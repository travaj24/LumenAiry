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

import ast
import json
import operator
import warnings
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
    """Sanitise a name for use as a filename.

    Path-hostile characters (``/``, ``\\``, spaces) are replaced with
    ``_``.  This mapping is many-to-one, so two *distinct* library names
    can sanitise to the SAME on-disk filename and silently clobber one
    another on save (audit S4-19).  When sanitisation actually changes
    the name we emit a ``UserWarning`` naming both forms so the
    collision risk is surfaced rather than silent.  (Python's default
    warning filter de-duplicates identical messages, so a repeated
    save/load of the same name warns only once.)
    """
    safe = name.replace('/', '_').replace('\\', '_').replace(' ', '_')
    if safe != name:
        warnings.warn(
            f"user_library: name {name!r} sanitised to {safe!r} for "
            f"on-disk storage; distinct names that sanitise to the same "
            f"filename collide and overwrite one another. Use a name "
            f"without '/', '\\', or spaces to avoid this.",
            UserWarning,
            stacklevel=2,
        )
    return safe


# =========================================================================
# Safe expression evaluator for phase-mask formulas (S4-19)
# =========================================================================
#
# Phase-mask "expression" masks let a user store an arbitrary formula
# (e.g. ``atan2(Y, X) * 3``) evaluated on the (X, Y) grid.  This used to
# be dispatched through the built-in ``eval()`` with the whole ``np``
# module exposed -- a code-execution risk if anyone can write to the
# library JSON (``eval("__import__('os').system(...)")`` etc.).  The
# ``__``/leading-underscore string screen in front of it was a leaky
# blocklist.
#
# ``_safe_eval_expression`` replaces that with a small ALLOWLIST AST
# interpreter: it parses the source in ``eval`` mode and evaluates the
# tree node-by-node, permitting ONLY arithmetic / comparison / boolean /
# bitwise operators, subscripts, whitelisted names, attribute access on
# a curated ``np`` math namespace, and calls to whitelisted callables.
# Anything else -- imports, lambdas, comprehensions, dunder / private
# attribute access, calls to non-whitelisted objects -- raises a clear
# ``ValueError``.  ``eval`` / ``exec`` / ``compile``-to-code are NEVER
# run on the parsed tree.

# Curated whitelist of PURE-MATH numpy attributes reachable as
# ``np.<name>`` from a phase-mask expression.  No I/O, no introspection,
# no object construction beyond elementwise math / small array helpers.
_NP_SAFE_ATTRS = frozenset({
    # constants
    'pi', 'e', 'euler_gamma', 'inf', 'nan', 'newaxis',
    # trigonometry
    'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'arctan2',
    'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh',
    'hypot', 'deg2rad', 'rad2deg', 'degrees', 'radians', 'unwrap',
    # exponentials / logs / powers
    'exp', 'exp2', 'expm1', 'log', 'log2', 'log10', 'log1p',
    'sqrt', 'cbrt', 'square', 'power', 'float_power', 'reciprocal',
    # rounding / sign / magnitude
    'abs', 'absolute', 'fabs', 'sign', 'floor', 'ceil', 'round',
    'rint', 'trunc', 'fix', 'mod', 'fmod', 'remainder',
    # clipping / extrema
    'clip', 'minimum', 'maximum', 'fmin', 'fmax',
    # complex
    'real', 'imag', 'conj', 'conjugate', 'angle',
    # elementwise selection / misc
    'where', 'heaviside', 'sinc', 'nan_to_num',
    # small array constructors sometimes used in masks
    'zeros_like', 'ones_like', 'full_like',
})

# Safe *data* attributes reachable on an array / scalar value.  All are
# non-callable, so ``.real``/``.imag``/``.T`` cannot reach the
# filesystem or introspection surface.
_ARRAY_SAFE_ATTRS = frozenset({'real', 'imag', 'T'})

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.BitAnd: operator.and_, ast.BitOr: operator.or_,
    ast.BitXor: operator.xor, ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos, ast.USub: operator.neg,
    ast.Invert: operator.invert, ast.Not: operator.not_,
}
_CMP_OPS = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne,
    ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
}


def _safe_eval_expression(expr: str, variables: Dict[str, Any]) -> Any:
    """Safely evaluate a phase-mask expression via a restricted AST walk.

    Parameters
    ----------
    expr : str
        The expression source (``eval``-mode).
    variables : dict
        The only permitted free names -- the mask grids (``X``, ``Y``,
        ``R``, ``THETA``, ``k``, ``pi``), the ``np`` module (attribute
        access restricted to :data:`_NP_SAFE_ATTRS`), and the curated
        math shortcuts (``sin``, ``cos``, ...).

    Raises
    ------
    ValueError
        If the expression fails to parse or contains any construct
        outside the allowlist (a name/attribute/call/operator/node type
        that is not permitted).
    """
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError as exc:
        raise ValueError(
            f"load_phase_mask: could not parse expression {expr!r}: {exc}"
        ) from exc

    def _ev(node):
        if isinstance(node, ast.Expression):
            return _ev(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (bool, int, float, complex)):
                return node.value
            raise ValueError(
                "load_phase_mask: a constant of type "
                f"{type(node.value).__name__!r} is not allowed in a "
                "phase-mask expression (only numeric literals).")

        if isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            raise ValueError(
                f"load_phase_mask: name {node.id!r} is not a permitted "
                "phase-mask variable. Allowed names: "
                f"{', '.join(sorted(variables))}.")

        if isinstance(node, ast.BinOp):
            op = _BIN_OPS.get(type(node.op))
            if op is None:
                raise ValueError(
                    "load_phase_mask: operator "
                    f"{type(node.op).__name__} is not allowed.")
            return op(_ev(node.left), _ev(node.right))

        if isinstance(node, ast.UnaryOp):
            op = _UNARY_OPS.get(type(node.op))
            if op is None:
                raise ValueError(
                    "load_phase_mask: unary operator "
                    f"{type(node.op).__name__} is not allowed.")
            return op(_ev(node.operand))

        if isinstance(node, ast.BoolOp):
            # Short-circuit like Python; on array operands numpy raises
            # the usual ambiguous-truth-value error (expected).
            if isinstance(node.op, ast.And):
                result = True
                for value in node.values:
                    result = _ev(value)
                    if not result:
                        return result
                return result
            result = False
            for value in node.values:
                result = _ev(value)
                if result:
                    return result
            return result

        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise ValueError(
                    "load_phase_mask: chained comparisons are not "
                    "supported; combine single comparisons with '&' / "
                    "'|' instead.")
            op = _CMP_OPS.get(type(node.ops[0]))
            if op is None:
                raise ValueError(
                    "load_phase_mask: comparison operator "
                    f"{type(node.ops[0]).__name__} is not allowed.")
            return op(_ev(node.left), _ev(node.comparators[0]))

        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith('_'):
                raise ValueError(
                    f"load_phase_mask: attribute access to {attr!r} is "
                    "not allowed (underscore / dunder access is blocked).")
            value = _ev(node.value)
            if value is np:
                if attr not in _NP_SAFE_ATTRS:
                    raise ValueError(
                        f"load_phase_mask: 'np.{attr}' is not in the "
                        "whitelisted numpy math namespace.")
                return getattr(np, attr)
            if attr in _ARRAY_SAFE_ATTRS:
                return getattr(value, attr)
            raise ValueError(
                f"load_phase_mask: attribute access '.{attr}' is not "
                "allowed.")

        if isinstance(node, ast.Call):
            if any(isinstance(a, ast.Starred) for a in node.args):
                raise ValueError(
                    "load_phase_mask: '*args' unpacking is not allowed.")
            func = _ev(node.func)
            if not callable(func):
                raise ValueError(
                    "load_phase_mask: attempted to call a non-callable "
                    "value.")
            args = [_ev(a) for a in node.args]
            kwargs = {}
            for kw in node.keywords:
                if kw.arg is None:
                    raise ValueError(
                        "load_phase_mask: '**kwargs' unpacking is not "
                        "allowed.")
                kwargs[kw.arg] = _ev(kw.value)
            return func(*args, **kwargs)

        if isinstance(node, ast.Subscript):
            return _ev(node.value)[_ev(node.slice)]

        if isinstance(node, ast.Slice):
            lower = _ev(node.lower) if node.lower is not None else None
            upper = _ev(node.upper) if node.upper is not None else None
            step = _ev(node.step) if node.step is not None else None
            return slice(lower, upper, step)

        if isinstance(node, ast.Tuple):
            return tuple(_ev(e) for e in node.elts)

        if isinstance(node, ast.List):
            return [_ev(e) for e in node.elts]

        raise ValueError(
            "load_phase_mask: expression element "
            f"{type(node).__name__} is not allowed in a phase-mask "
            "formula.")

    return _ev(tree)


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
       k (wavenumber), pi.  The formula is evaluated by a restricted
       allowlist AST interpreter (S4-19), not Python ``eval``: pure-math
       numpy functions are available via ``np.<func>`` or the shortcut
       names (``sin``, ``cos``, ``tan``, ``sqrt``, ``abs``, ``exp``,
       ``log``, ``atan2``, ``mod``, ``floor``, ``ceil``); imports,
       lambdas, comprehensions, and attribute/introspection escapes are
       rejected.

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

        # S4-19: the phase-mask expression is a code-execution risk if
        # anyone can write to ~/.lumenairy/library/phase_masks/*.json.
        # Evaluate it through the restricted allowlist AST interpreter
        # (:func:`_safe_eval_expression`) -- NOT the built-in ``eval`` --
        # so only arithmetic / comparison / boolean / bitwise ops,
        # subscripts, whitelisted names, and calls to the curated numpy
        # math namespace are permitted; imports, lambdas, comprehensions,
        # and dunder / private attribute access all raise ``ValueError``.
        expr = str(data['expression'])
        ns = {
            'X': X, 'Y': Y, 'R': R, 'THETA': THETA,
            'k': k, 'pi': pi, 'np': np,
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'sqrt': np.sqrt, 'abs': np.abs,
            'exp': np.exp, 'log': np.log,
            'atan2': np.arctan2, 'arctan2': np.arctan2,
            'mod': np.mod, 'floor': np.floor, 'ceil': np.ceil,
        }
        phase = _safe_eval_expression(expr, ns)
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

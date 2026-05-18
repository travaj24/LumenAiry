# Cluster B Implementation Spec — Operator Algebra + Rays-from-Field

**Status**: Specification draft for AI agent implementation
**Target version**: LumenAiry 4.15.x (additive features, no API breakage)
**Scope**: Two additive features (Item 2 + Item 6). Item 3 (`suggest_grid`) is **already implemented** — see §6.

---

## 1. Background

This work originated from a comparative review of a small in-house repo (`Neurophos/lens-designer`, ~4.7 kLOC, frozen at "initial code") against the current LumenAiry codebase. The lens-designer repo carried a Nazarathy/Shamir-style operator algebra (`Q`, `F`, `V`, `R` classes with `__mul__` composition and automatic ABCD tracking) plus a coherent-field-to-rays bridge. Neither of those exists in LumenAiry, and both are **additive** — they can sit on top of LumenAiry's existing array-first propagator infrastructure without forking the field representation or breaking any existing API.

A larger architectural change ("Cluster A" — Σ-matrix phase-space grid evolution, `Field` as callable closure, automatic `build_grid` sizing) was considered and **rejected**: it would require a parallel field-representation API, only applies in the paraxial regime, and its better non-paraxial cousin (Σ-evolution applied to *beamlets*) already exists as `lumenairy/propagators/gbd.py`. This spec covers only Cluster B.

### What you (the implementing agent) should NOT do

- **Do not** introduce a `Field` callable-closure class. Fields are `numpy.ndarray` in LumenAiry; the `Source` dataclass at [`sources/core.py:1749`](lumenairy/sources/core.py) wraps `(E, dx, wavelength, dy)`. Keep it that way.
- **Do not** introduce a `Grid` class that tracks Σ-matrix covariance or `fov_safety` margin. Sampling stays as explicit `(N, dx, dy)`.
- **Do not** replace `propagate_through_system` ([`system.py:47`](lumenairy/system.py)) or `propagate` ([`propagators/dispatch.py:43`](lumenairy/propagators/dispatch.py)) — the new operator-algebra surface sits *next to* those, not in front of them.
- **Do not** reimplement grid sizing — `recommend_grid_for_prescription` ([`elements/lenses.py:445`](lumenairy/elements/lenses.py)) already does it.
- **Do not** reimplement ABCD computation — `system_abcd` and `lens_abcd` in [`raytrace/core.py:2194`](lumenairy/raytrace/core.py) are the source-of-truth for ground-truth ABCDs in tests.
- **Do not** alter `RayBundle`'s schema (see [`raytrace/core.py:65`](lumenairy/raytrace/core.py)). Use it as-is.

---

## 2. Existing infrastructure (read before writing code)

These already exist and the new code must integrate with them, not replicate them.

| Symbol | Location | Role for this spec |
|---|---|---|
| `Source` dataclass | [`sources/core.py:1749`](lumenairy/sources/core.py) | Carries `(E, dx, dy, wavelength)`. Operators in Item 2 should accept and return `Source` (and also a bare `(E, dx, wavelength)` form). |
| `RayBundle` dataclass | [`raytrace/core.py:65`](lumenairy/raytrace/core.py) | Per-component arrays `(x, y, z, L, M, N, wavelength, alive, opd, error_code)`. Output type of Item 6. |
| `RAY_OK`, `RAY_EVANESCENT`, etc. | [`raytrace/core.py:55-62`](lumenairy/raytrace/core.py) | Use `RAY_EVANESCENT` for rays whose `L² + M² > 1`. |
| `system_abcd(surfaces, wavelength)` | [`raytrace/core.py:2194`](lumenairy/raytrace/core.py) | Returns `(M, efl, bfl, ffl)` for a Surface list. **Ground truth** for Item 2's ABCD composition tests. |
| `lens_abcd(...)` | [`raytrace/core.py:2381`](lumenairy/raytrace/core.py) | Per-lens ABCD. Useful for `Operator.from_lens`. |
| `surfaces_from_prescription(prescription)` | [`raytrace/core.py`](lumenairy/raytrace/core.py) | Prescription → Surface list. Used by `Operator.from_prescription`. |
| `angular_spectrum_propagate(E, z, wavelength, dx, ...)` | [`propagators/propagation.py:1596`](lumenairy/propagators/propagation.py) | Delegation target for `FreeSpace.__call__`. |
| `fresnel_propagate`, `fraunhofer_propagate`, `rayleigh_sommerfeld_propagate`, `scalable_angular_spectrum_propagate` | [`propagators/propagation.py`](lumenairy/propagators/propagation.py) | Alternate delegation targets selectable by `method=` kwarg. |
| `propagate(E_in, ...)` smart dispatcher | [`propagators/dispatch.py:43`](lumenairy/propagators/dispatch.py) | Use this as the default backend in `FreeSpace.__call__` (`method='auto'`). |
| `apply_thin_lens(E, f, wavelength, dx, ...)` | [`elements/lenses.py`](lumenairy/elements/lenses.py) | Delegation target for `ThinLens.__call__`. |
| `apply_cylindrical_lens` | [`elements/lenses.py`](lumenairy/elements/lenses.py) | Delegation target when cylindrical ABCD is requested. |
| `apply_aperture`, `apply_gaussian_aperture` | [`elements/`](lumenairy/elements/) | Delegation target for `Aperture.__call__`. |
| `recommend_grid_for_prescription` | [`elements/lenses.py:445`](lumenairy/elements/lenses.py) | Item 3 already-exists answer. |
| `propagate_through_system` | [`system.py:47`](lumenairy/system.py) | Existing prescription-driven workflow. Leave alone. The operator algebra is a parallel surface for users who think algebraically. |

### Conventions to follow

- **Time convention**: `exp(-iωt)`. Already documented at [`lumenairy/__init__.py:8`](lumenairy/__init__.py).
- **Units**: SI throughout. Wavelength is in **vacuum** meters. Lengths in meters.
- **Anamorphic support**: separate `dx` and `dy`. Operators must track separate ABCDs for x and y (2 × 2 each). Default `dy = dx` for square grids.
- **Author tag**: Each new module should include `Author: Andrew Traverso` in the header (matches existing convention).
- **Audit posture**: this codebase has rounds of audit-driven P0/P1/P2 fixes recorded in `CHANGELOG.md`. New code should be *defensive*: validate inputs at boundaries, fail loudly on inconsistent units, document the math explicitly in docstrings, never silently downgrade precision. Mirror the docstring style of `system_abcd` and `angular_spectrum_propagate`.
- **Backwards compatibility**: every public function gets `return_result: bool = False` if it returns a numpy array primarily but can optionally wrap in a result object. Don't change return types of existing functions.
- **Public exports**: add new symbols to `lumenairy/__init__.py` and `lumenairy/<subpackage>/__init__.py`. Maintain alphabetical order within sections. Add to the `__all__` list at bottom of `lumenairy/__init__.py`.

---

## 3. Item 2 — Operator Algebra Layer

### 3.1 Goal

Provide a Python surface for Nazarathy/Shamir-style optical operator algebra:

```python
import lumenairy as la
import numpy as np

# Build a 4f system symbolically
f = 100e-3
sys = (
    la.algebra.FreeSpace(f)
    * la.algebra.ThinLens(f)
    * la.algebra.FreeSpace(2 * f)
    * la.algebra.ThinLens(f)
    * la.algebra.FreeSpace(f)
)

# Inspect the system ABCD (no field needed)
print(sys.abcd)             # [[-1, 0], [0, -1]]  — 4f is an inverter
print(sys.efl)              # +inf (afocal)

# Apply it to a Source
src = la.Source.gaussian(N=512, dx=1e-6, wavelength=633e-9, w0=100e-6)
out = sys(src)              # Source -> Source, propagates through every stage

# Or apply it to a bare ndarray
E_out = sys.apply(src.E, dx=src.dx, wavelength=src.wavelength)
```

This gives users:
1. Ability to read the system ABCD off a composed expression without ever applying it to a field.
2. A symbolic-construction surface that matches the way the Nazarathy/Shamir literature writes optical systems.
3. Reusability — define `sys` once, apply to many sources.

The execution model is **chain-and-delegate**: composition multiplies ABCDs; application walks the chain right-to-left applying each operator's underlying LumenAiry function. No closure-based field representation; no symbolic reduction (Phase 2 only — see §3.6).

### 3.2 File layout

Create a new subpackage:

```
lumenairy/algebra/
├── __init__.py            # public exports
├── base.py                # Operator, CompositeOperator, ABCD helpers
├── primitives.py          # FreeSpace, ThinLens, CylindricalLens, Magnify, FourierTransform
├── apertures.py           # Aperture, GaussianAperture
├── from_prescription.py   # Operator.from_prescription factory + Surface-list bridge
└── __init__.py
```

Add to `lumenairy/__init__.py`:
```python
from .algebra import (
    Operator,
    CompositeOperator,
    FreeSpace,
    ThinLens,
    CylindricalLens,
    Magnify,
    FourierTransform,
    Aperture,
    GaussianAperture,
)
```

### 3.3 API surface

#### 3.3.1 `Operator` base class (in `base.py`)

```python
class Operator:
    """Base class for composable optical operators in the
    Nazarathy/Shamir algebraic formalism.

    Each Operator carries 2×2 ABCD matrices for the x and y directions
    (real-valued, supports anamorphic systems) and implements an
    `_apply(E, dx, dy, wavelength)` hook that delegates to existing
    LumenAiry propagators/element functions.

    Operators compose via `*` (matrix multiply on ABCDs, chain of
    `_apply` calls right-to-left) and apply to a field via `__call__`.

    Convention follows Nazarathy & Shamir, "Fourier optics described
    by operator algebra," JOSA 70 (2), 1980.
    """

    # Subclasses set these in __init__:
    _abcd_x: np.ndarray  # shape (2, 2), real, float64
    _abcd_y: np.ndarray  # shape (2, 2), real, float64

    # ABCD readout
    @property
    def abcd(self) -> np.ndarray:
        """System ABCD if isotropic; raises ValueError if anamorphic."""
        if np.allclose(self._abcd_x, self._abcd_y):
            return self._abcd_x.copy()
        raise ValueError("Anamorphic system — use abcd_x/abcd_y instead.")

    @property
    def abcd_x(self) -> np.ndarray: ...
    @property
    def abcd_y(self) -> np.ndarray: ...

    @property
    def A(self) -> float: ...      # element accessors, with anamorphic guard
    @property
    def B(self) -> float: ...
    @property
    def C(self) -> float: ...
    @property
    def D(self) -> float: ...

    @property
    def efl(self) -> float:
        """Effective focal length = -1/C; +inf if C == 0 (afocal)."""

    @property
    def is_anamorphic(self) -> bool:
        return not np.allclose(self._abcd_x, self._abcd_y)

    # Composition
    def __mul__(self, other: "Operator") -> "CompositeOperator":
        """A * B means "first B, then A" (matrix-on-the-left convention)."""

    # Application
    def __call__(self, source: Union[Source, Tuple]) -> Union[Source, Tuple]:
        """Apply to a Source (returns Source) or (E, dx, wavelength) tuple
        (returns tuple). Anamorphic-aware (passes dy through)."""

    def apply(self, E: np.ndarray, *, dx: float, wavelength: float,
              dy: Optional[float] = None) -> np.ndarray:
        """Bare-ndarray entry point. Returns E_out (complex ndarray)."""

    # Subclass hook
    def _apply(self, E: np.ndarray, *, dx: float, dy: float,
               wavelength: float) -> Tuple[np.ndarray, float, float]:
        """Apply this operator to a complex field. Returns
        (E_out, dx_out, dy_out). dx/dy may change (e.g. FreeSpace via
        Fresnel changes pitch). Subclasses MUST override.
        """
        raise NotImplementedError
```

**Critical convention**: `A * B` means "apply B first, then A". This matches the matrix-on-the-left convention used in `system_abcd` ([`raytrace/core.py:2237-2290`](lumenairy/raytrace/core.py)) where `M = R_mat @ M` after each surface. The ABCD of `A * B` is `A.abcd @ B.abcd`. Application order is `A(B(field))`.

#### 3.3.2 `CompositeOperator` (in `base.py`)

```python
class CompositeOperator(Operator):
    """A chain of operators. Stores the original chain so we can
    inspect intermediate stages; also stores the precomputed product
    ABCD for fast readout.

    Constructed implicitly via `op_a * op_b`. Users normally don't
    instantiate directly.
    """
    def __init__(self, chain: List[Operator]):
        # chain[0] is applied first, chain[-1] last (left-to-right read,
        # right-to-left math: chain[-1] * ... * chain[1] * chain[0])
        self._chain = list(chain)
        # Recompute composite ABCD
        self._abcd_x = np.eye(2)
        self._abcd_y = np.eye(2)
        for op in chain:
            self._abcd_x = op._abcd_x @ self._abcd_x
            self._abcd_y = op._abcd_y @ self._abcd_y

    def _apply(self, E, *, dx, dy, wavelength):
        for op in self._chain:
            E, dx, dy = op._apply(E, dx=dx, dy=dy, wavelength=wavelength)
        return E, dx, dy

    def stages(self) -> List[Operator]:
        """Read-only view of the underlying chain (for debugging / viz)."""
        return list(self._chain)

    def apply_with_intermediates(self, E, *, dx, wavelength, dy=None):
        """Apply the chain, returning a list of (E, dx, dy) tuples
        after each stage. Useful for through-stage visualization."""
        ...
```

Composition flattens nested `CompositeOperator`s:
```python
def __mul__(self, other):
    left_chain = self._chain if isinstance(self, CompositeOperator) else [self]
    right_chain = other._chain if isinstance(other, CompositeOperator) else [other]
    return CompositeOperator(right_chain + left_chain)  # right applied first
```

#### 3.3.3 Primitive operators (in `primitives.py`)

| Class | ABCD (x and y both, unless noted) | `_apply` delegates to |
|---|---|---|
| `FreeSpace(d, *, method='auto')` | `[[1, d], [0, 1]]` | `propagators.dispatch.propagate(E, z=d, ..., method=method)` |
| `ThinLens(f)` | `[[1, 0], [-1/f, 1]]` | `elements.lenses.apply_thin_lens(E, f, wavelength, dx)` |
| `CylindricalLens(f_x=inf, f_y=inf)` | `x: [[1,0],[-1/f_x, 1]]`, `y: [[1,0],[-1/f_y,1]]` | `elements.lenses.apply_cylindrical_lens(E, ...)` |
| `Magnify(a_x, a_y=None)` | `[[1/a, 0], [0, a]]` (anamorphic if a_x ≠ a_y) | `propagators.propagation.resample_field(E, ...)` with grid-pitch rescale |
| `FourierTransform(f_focal)` | `[[0, -f_focal/k0], [k0/f_focal, 0]]` — see note | Two `FreeSpace(f_focal)` + one `ThinLens(f_focal)` (2f setup) OR direct FFT with proper normalization, see §3.4 |
| `Aperture(diameter, shape='circular')` | identity (`[[1,0],[0,1]]`) | `elements.apply_aperture(E, ...)` |
| `GaussianAperture(sigma)` | identity | `elements.apply_gaussian_aperture(E, ...)` |

Notes:

- `FreeSpace.method` defaults to `'auto'` and forwards to the dispatch layer; explicit settings `'asm'`, `'fresnel'`, `'sas'`, `'rs'`, `'fraunhofer'` are honored.
- `ThinLens(f)` for `f = np.inf` returns identity-ABCD (no power).
- `Magnify(a)` corresponds to Nazarathy/Shamir's `V[a]` with the **corrected** ABCD `diag(1/a, a)`. Conserves energy via the `√a` amplitude prefactor in `_apply` (see [lens-designer operators.py:556-577](file:///lens-designer/operators.py) for the reference implementation).
- `FourierTransform(f_focal)` is the **physical** optical Fourier transform, realized at the back focal plane of a thin lens with focal length `f_focal`. The pure mathematical Fourier transform without a focal length is dimensionally nonsensical in real units; we deliberately require `f_focal`. Internally implement as either (a) Fresnel propagation by `f_focal` with the lens phase folded in, or (b) a `FreeSpace(f) * ThinLens(f) * FreeSpace(f)` composite (cleaner but slower). Pick option (a) for performance.

#### 3.3.4 `Operator.from_prescription` factory

```python
class Operator:
    @classmethod
    def from_prescription(
        cls,
        prescription: Dict[str, Any],
        wavelength: float,
        *,
        method: str = 'auto',
    ) -> "CompositeOperator":
        """Build a CompositeOperator from a LumenAiry prescription dict.

        Walks the prescription's surface list (via
        `surfaces_from_prescription`), folding each refractive surface's
        contribution to ABCD and each air-gap thickness into a
        `FreeSpace`. Mirror surfaces flip the propagation sign per the
        Welford convention (matches `system_abcd`'s mirror_parity).

        The resulting CompositeOperator has `abcd` exactly equal to
        the matrix returned by `system_abcd(surfaces, wavelength)` —
        this is asserted in tests.

        For execution, each refracting surface becomes a `ThinLens`
        with `f = 1 / phi` where `phi = (n2 - n1) / R` is the surface
        power, and each thickness becomes a `FreeSpace(t / n_after)`
        (reduced thickness). This is paraxial-only — for high-NA
        prescriptions users should call `propagate_through_system`
        directly with the original prescription rather than the
        operator-algebra path.
        """
```

A warning on `from_prescription`: the operator-algebra path is **paraxial-only**. For real-lens / freeform / high-NA prescriptions, the user should keep using `propagate_through_system` and the operator-algebra path is for analysis/sanity-checking, not production. Document this loudly in the docstring.

### 3.4 ABCD math reference

For tests, every primitive's ABCD must be exactly:

```python
# FreeSpace(d)
M = np.array([[1, d], [0, 1]])

# ThinLens(f)
M = np.array([[1, 0], [-1/f, 1]])    # f = +inf -> identity

# CylindricalLens(f_x, f_y) — separate per axis
M_x = np.array([[1, 0], [-1/f_x, 1]])
M_y = np.array([[1, 0], [-1/f_y, 1]])

# Magnify(a)   (Nazarathy/Shamir V[a] with corrected diag(1/a, a))
M = np.array([[1/a, 0], [0, a]])

# FourierTransform(f_focal)
# ABCD of the 2f setup [FreeSpace(f) * ThinLens(f) * FreeSpace(f)]:
M = np.array([[0, f_focal], [-1/f_focal, 0]])

# Aperture, GaussianAperture
M = np.eye(2)
```

These should be unit-tested literally — see §3.7.

Composition: `(A * B).abcd == A.abcd @ B.abcd`.

EFL: `efl = -1.0 / C` if `abs(C) > 1e-30` else `+inf`. Matches `system_abcd`'s convention.

### 3.5 Application delegation table

Each primitive's `_apply(E, dx, dy, wavelength)` does:

```python
class FreeSpace(Operator):
    def _apply(self, E, *, dx, dy, wavelength):
        from ..propagators.dispatch import propagate
        E_out = propagate(E, z=self.distance, wavelength=wavelength,
                          dx=dx, method=self.method)
        # propagate may return a tuple (E, dx_out, dy_out) for Fresnel/SAS.
        # Coerce via the same logic dispatch.py uses internally
        # (see _coerce_field at dispatch.py:169).
        return _coerce_propagation_output(E_out, dx_default=dx, dy_default=dy)

class ThinLens(Operator):
    def _apply(self, E, *, dx, dy, wavelength):
        from ..elements.lenses import apply_thin_lens
        if np.isinf(self.f):
            return E, dx, dy
        E_out = apply_thin_lens(E, f=self.f, wavelength=wavelength, dx=dx)
        return E_out, dx, dy

class CylindricalLens(Operator):
    def _apply(self, E, *, dx, dy, wavelength):
        from ..elements.lenses import apply_cylindrical_lens
        E_out = apply_cylindrical_lens(
            E, f_x=self.f_x, f_y=self.f_y,
            wavelength=wavelength, dx=dx, dy=dy,
        )
        return E_out, dx, dy

class Magnify(Operator):
    def _apply(self, E, *, dx, dy, wavelength):
        from ..propagators.propagation import resample_field
        # Rescale grid pitch: new pitch = old / a
        new_dx = dx / self.a_x
        new_dy = dy / self.a_y
        E_out = resample_field(E, dx_in=dx, dx_out=new_dx,
                                dy_in=dy, dy_out=new_dy)
        # Amplitude prefactor for energy conservation:
        # |E_out|^2 * dx_out * dy_out == |E_in|^2 * dx_in * dy_in
        E_out *= np.sqrt(self.a_x * self.a_y)
        return E_out, new_dx, new_dy

class FourierTransform(Operator):
    def _apply(self, E, *, dx, dy, wavelength):
        # Option (a): apply lens phase then Fresnel-propagate by f.
        from ..elements.lenses import apply_thin_lens
        from ..propagators.propagation import fresnel_propagate
        E1 = apply_thin_lens(E, f=self.f, wavelength=wavelength, dx=dx)
        E_out, dx_out, dy_out = fresnel_propagate(
            E1, z=self.f, wavelength=wavelength, dx=dx, dy=dy,
        )
        return E_out, dx_out, dy_out

class Aperture(Operator):
    def _apply(self, E, *, dx, dy, wavelength):
        from ..elements import apply_aperture
        E_out = apply_aperture(E, diameter=self.diameter,
                                shape=self.shape, dx=dx, dy=dy)
        return E_out, dx, dy
```

`CompositeOperator._apply` just walks the chain:
```python
def _apply(self, E, *, dx, dy, wavelength):
    for op in self._chain:
        E, dx, dy = op._apply(E, dx=dx, dy=dy, wavelength=wavelength)
    return E, dx, dy
```

### 3.6 Symbolic reduction (Phase 2, OPTIONAL)

After the chain-and-delegate version is shipped and tested, a follow-up PR may add **symbolic reduction**: detect canonical operator-algebra patterns and collapse them into a single closed-form operation. Examples from Nazarathy/Shamir §IV:

- `FreeSpace(d) * ThinLens(f) * FreeSpace(d)` with `d = f` → folded into a single `FourierTransform(f)` (one FFT instead of three propagations).
- `ThinLens(f1) * ThinLens(f2)` → `ThinLens(1/(1/f1 + 1/f2))` (combined power).
- `FreeSpace(d1) * FreeSpace(d2)` → `FreeSpace(d1 + d2)`.
- Any chain reducible to a Collins integral / Q-F-Q sandwich.

**Do not implement Phase 2 in the first PR.** It's a performance optimization, not a correctness requirement. The chain-and-delegate version is functionally complete; symbolic reduction is gravy. Open a separate roadmap item.

### 3.7 Tests for Item 2

Add to `tests/test_algebra/`:

#### `test_algebra_abcd.py`
Pure ABCD readout, no field application:

```python
def test_freespace_abcd():
    op = FreeSpace(0.1)
    assert np.allclose(op.abcd, [[1.0, 0.1], [0.0, 1.0]])
    assert np.isinf(op.efl)  # afocal

def test_thinlens_abcd():
    op = ThinLens(0.05)
    assert np.allclose(op.abcd, [[1.0, 0.0], [-20.0, 1.0]])
    assert np.isclose(op.efl, 0.05)

def test_thinlens_infinite_focal_length_is_identity():
    op = ThinLens(np.inf)
    assert np.allclose(op.abcd, np.eye(2))

def test_composition_matches_matmul():
    # 4f imaging system
    f = 0.1
    sys = (FreeSpace(f) * ThinLens(f) * FreeSpace(2*f)
            * ThinLens(f) * FreeSpace(f))
    expected = (np.array([[1, f], [0, 1]])
                @ np.array([[1, 0], [-1/f, 1]])
                @ np.array([[1, 2*f], [0, 1]])
                @ np.array([[1, 0], [-1/f, 1]])
                @ np.array([[1, f], [0, 1]]))
    assert np.allclose(sys.abcd, expected)
    # 4f inverter: ABCD = [[-1, 0], [0, -1]]
    assert np.allclose(sys.abcd, [[-1, 0], [0, -1]], atol=1e-10)

def test_composition_associativity():
    a, b, c = FreeSpace(0.01), ThinLens(0.05), FreeSpace(0.02)
    assert np.allclose(((a * b) * c).abcd, (a * (b * c)).abcd)

def test_anamorphic_cylindrical_lens():
    op = CylindricalLens(f_x=np.inf, f_y=0.05)
    assert op.is_anamorphic
    assert np.allclose(op.abcd_x, np.eye(2))
    assert np.allclose(op.abcd_y, [[1, 0], [-20, 1]])
```

#### `test_algebra_matches_system_abcd.py`
Cross-check against existing `system_abcd`:

```python
def test_from_prescription_matches_system_abcd():
    """For a paraxial prescription, Operator.from_prescription must
    produce an ABCD identical to system_abcd(surfaces, wavelength)."""
    from lumenairy.raytrace.core import (
        surfaces_from_prescription, system_abcd,
    )
    prescription = {
        # ... a representative singlet + air gap + doublet prescription ...
    }
    wavelength = 633e-9
    surfaces = surfaces_from_prescription(prescription)
    M_ref, efl_ref, _, _ = system_abcd(surfaces, wavelength)

    op = Operator.from_prescription(prescription, wavelength)
    assert np.allclose(op.abcd, M_ref, atol=1e-12)
    assert np.isclose(op.efl, efl_ref)
```

Use prescriptions from existing tests as fixtures (look at `tests/` for prior prescription fixtures).

#### `test_algebra_application.py`
Field-application correctness:

```python
def test_freespace_application_matches_dispatcher():
    """FreeSpace(d)(source) must produce the same E as
    propagate(source.E, z=d, ...)."""
    src = Source.gaussian(N=256, dx=2e-6, wavelength=633e-9, w0=50e-6)
    op = FreeSpace(0.05, method='asm')
    out = op(src)
    from lumenairy import angular_spectrum_propagate
    E_ref = angular_spectrum_propagate(
        src.E, z=0.05, wavelength=src.wavelength, dx=src.dx,
    )
    assert np.allclose(out.E, E_ref)

def test_thinlens_application_matches_apply_thin_lens():
    src = Source.gaussian(N=256, dx=2e-6, wavelength=633e-9, w0=50e-6)
    op = ThinLens(0.1)
    out = op(src)
    from lumenairy import apply_thin_lens
    E_ref = apply_thin_lens(
        src.E, f=0.1, wavelength=src.wavelength, dx=src.dx,
    )
    assert np.allclose(out.E, E_ref)

def test_composite_application_matches_sequential():
    """sys(src) for sys = A * B * C must equal A(B(C(src)))."""
    src = Source.gaussian(N=256, dx=2e-6, wavelength=633e-9, w0=50e-6)
    a, b, c = FreeSpace(0.01), ThinLens(0.05), FreeSpace(0.05)
    sys = a * b * c
    out_composite = sys(src)
    out_sequential = a(b(c(src)))
    assert np.allclose(out_composite.E, out_sequential.E)

def test_4f_inverter():
    """A 4f system [f, lens(f), 2f, lens(f), f] should image a Gaussian
    centered at +x0 to a Gaussian centered at -x0 with unit
    magnification."""
    f = 0.1
    x0 = 30e-6
    src = Source.gaussian(N=512, dx=1e-6, wavelength=633e-9, w0=20e-6, x0=x0)
    sys = (FreeSpace(f) * ThinLens(f) * FreeSpace(2*f)
            * ThinLens(f) * FreeSpace(f))
    out = sys(src)
    # Find centroid of |out.E|^2
    intensity = np.abs(out.E)**2
    ix = np.arange(out.E.shape[1])
    centroid_idx = np.sum(ix * intensity.sum(axis=0)) / intensity.sum()
    centroid_x = (centroid_idx - out.E.shape[1] // 2) * out.dx
    assert np.isclose(centroid_x, -x0, atol=2e-6)  # one pixel tolerance
```

#### `test_algebra_examples.py`
Integration: a few end-to-end mini-examples that exercise the API as a user would, including printing `op.abcd`, reading `op.efl`, etc.

### 3.8 Examples for Item 2

Add `examples/algebra_4f_system.py`:
```python
"""4f imaging system constructed via operator algebra.

Shows how to build a system symbolically, inspect its ABCD, and apply
it to a coherent Gaussian source. Compares the algebraic result to
the equivalent prescription-driven propagate_through_system call.
"""
import numpy as np
import lumenairy as la
import matplotlib.pyplot as plt

f = 100e-3
sys = (la.FreeSpace(f) * la.ThinLens(f) * la.FreeSpace(2*f)
       * la.ThinLens(f) * la.FreeSpace(f))

print(f"System ABCD: {sys.abcd}")            # [[-1, 0], [0, -1]]
print(f"EFL: {sys.efl}")                       # inf (afocal)
print(f"Magnification: {-sys.A}")               # +1 (inverted but unit mag)

src = la.Source.gaussian(N=512, dx=2e-6, wavelength=633e-9, w0=80e-6)
out = sys(src)

# Visualize input vs output intensity
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(np.abs(src.E)**2, cmap='hot')
axes[0].set_title('Input')
axes[1].imshow(np.abs(out.E)**2, cmap='hot')
axes[1].set_title('After 4f (inverted)')
plt.show()
```

Add `examples/algebra_anamorphic.py`: same idea but with a cylindrical lens cleaning up an astigmatic beam.

### 3.9 Acceptance criteria for Item 2

- All tests in §3.7 pass.
- `Operator.from_prescription(p, lam).abcd` matches `system_abcd(surfaces_from_prescription(p), lam)[0]` to within `1e-12` absolute tolerance for at least 3 representative prescriptions (singlet, doublet, telephoto).
- No existing test in `tests/` breaks.
- New module passes the existing audit-defensive style: input validation, clear error messages on shape/unit mismatch.
- All new symbols are exported via `lumenairy/__init__.py` and `lumenairy/algebra/__init__.py`.
- `examples/algebra_4f_system.py` runs to completion and produces a meaningful figure.
- Public docstrings cite Nazarathy & Shamir JOSA 1980 and link to `system_abcd` for the ABCD source-of-truth relationship.

---

## 4. Item 6 — Rays from Coherent Field

### 4.1 Goal

A bridge function that takes a coherent field and returns a `RayBundle` whose rays represent the field's geometric content. Bridges `propagators/` (wave) ↔ `raytrace/` (ray) so users can overlay ray traces on coherent-field plots, seed a Maslov / GBD bundle from a measured pupil field, or hand a coherent field into the geometric ray tracer for hybrid analysis.

This is **additive**: no existing API changes.

### 4.2 File location

Create a new module:

```
lumenairy/raytrace/from_field.py
```

Export from `lumenairy/raytrace/__init__.py` and `lumenairy/__init__.py`:
```python
from .raytrace import rays_from_field
```

### 4.3 API surface

```python
def rays_from_field(
    E: np.ndarray,
    *,
    dx: float,
    wavelength: float,
    dy: Optional[float] = None,
    n_rays: int = 200,
    placement: str = 'cdf',
    angle_method: str = 'complex_gradient',
    intensity_threshold: float = 1e-4,
    z0: float = 0.0,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> RayBundle:
    """Sample a coherent field into a geometric RayBundle.

    Each ray's position is drawn from the field's intensity
    distribution; each ray's direction comes from the local phase
    gradient k_⊥ = ∇φ via the paraxial mapping (L, M) = (k_x, k_y) / k0,
    N = sqrt(1 - L² - M²). Rays whose tangential k exceeds the
    wavenumber (evanescent components) are flagged with
    RAY_EVANESCENT and marked alive=False.

    The OPD is initialized to phi(x, y) / k0 so subsequent
    geometric-ray accumulation continues from the wave-optical phase.

    Parameters
    ----------
    E : ndarray, complex, shape (Ny, Nx)
        Coherent field amplitude.  Must be 2-D.
    dx : float
        Grid pitch in x [m].
    wavelength : float
        Vacuum wavelength [m].
    dy : float, optional
        Grid pitch in y [m].  Defaults to dx (square grid).
    n_rays : int, default 200
        Target number of rays.  May be fewer after intensity
        thresholding and evanescent filtering.
    placement : {'cdf', 'rejection', 'uniform'}
        Strategy for placing ray origins.

        - 'cdf' : separable inverse-CDF along x and y marginals.
          Fast; matches the lens-designer reference.  Approximate for
          non-separable intensity distributions but accurate enough
          for visualization.
        - 'rejection' : true 2-D rejection sampling from |E|².  Slower
          but exact.  Use when intensity is strongly non-separable
          (e.g. spiral phase plates, complicated holograms).
        - 'uniform' : place on a uniform grid, drop pixels below
          intensity_threshold.  Use when you want a regular ray fan
          weighted by survival.

    angle_method : {'complex_gradient', 'unwrap_gradient'}
        How to compute the local k-vector.

        - 'complex_gradient' (default) : k_⊥ = Im(∇E / E).
          Avoids phase unwrapping entirely; safe for fields with
          phase singularities.  Clamps |E| < intensity_threshold to
          intensity_threshold * max(|E|) to prevent division blowup.
        - 'unwrap_gradient' : k_⊥ = ∇(unwrap(angle(E))) via numpy's
          unwrap_2d.  Fragile near vortices; use only for smooth
          phase profiles.

    intensity_threshold : float, default 1e-4
        Relative threshold |E|² / max(|E|²).  Pixels below threshold
        are excluded from placement and used to clamp the
        complex-gradient denominator.
    z0 : float, default 0.0
        Axial position of the resulting ray origins.
    random_state : int | np.random.Generator | None
        Seed or generator for placement randomness.  None uses the
        default rng.  Set an int for reproducibility.

    Returns
    -------
    RayBundle
        Bundle with n_rays_actual <= n_rays rays.  Per-ray fields:
        - x, y : sampled origin coordinates [m]
        - z : np.full(n, z0) [m]
        - L, M, N : direction cosines, |L²+M²+N²| = 1
        - alive : True for non-evanescent rays, False otherwise
        - opd : phase(E_at_origin) / k0  [m]
        - error_code : RAY_OK or RAY_EVANESCENT
        - wavelength : wavelength

    Examples
    --------
    Seed a ray bundle from a measured pupil field for downstream
    geometric trace through a relay:

    >>> rays = rays_from_field(E_pupil, dx=2e-6, wavelength=633e-9,
    ...                        n_rays=500, placement='rejection')
    >>> # Now propagate rays through downstream optics via raytrace
    >>> rays_at_image = trace_through(rays, surfaces)

    Overlay a fan of rays on a coherent-field z-stack:

    >>> rays = rays_from_field(E_exit, dx=dx, wavelength=lam, n_rays=50)
    >>> # plot rays.x, rays.y vs z extended via L/M/N
    """
```

### 4.4 Math reference

**Placement, `'cdf'` mode** (separable inverse-CDF, fast):

```python
def _place_cdf(E, dx, dy, n_rays, threshold, rng):
    I = np.abs(E)**2
    I = I / I.max()
    # x-marginal
    Ix = I.sum(axis=0)
    Ix = np.where(Ix > threshold, Ix, 0)
    cdf_x = np.cumsum(Ix); cdf_x /= cdf_x[-1]
    # y-marginal
    Iy = I.sum(axis=1)
    Iy = np.where(Iy > threshold, Iy, 0)
    cdf_y = np.cumsum(Iy); cdf_y /= cdf_y[-1]
    # Draw uniforms, invert CDFs (independent because separable)
    u = rng.random(n_rays); v = rng.random(n_rays)
    ix = np.searchsorted(cdf_x, u)
    iy = np.searchsorted(cdf_y, v)
    Ny, Nx = E.shape
    x = (ix - Nx // 2) * dx
    y = (iy - Ny // 2) * dy
    return x, y, ix, iy
```

**Placement, `'rejection'` mode** (true 2-D, exact):

```python
def _place_rejection(E, dx, dy, n_rays, threshold, rng):
    I = np.abs(E)**2; I = I / I.max()
    Ny, Nx = E.shape
    x, y, ix_arr, iy_arr = [], [], [], []
    max_tries = n_rays * 50
    tries = 0
    while len(x) < n_rays and tries < max_tries:
        ix = rng.integers(0, Nx)
        iy = rng.integers(0, Ny)
        if I[iy, ix] >= threshold and rng.random() < I[iy, ix]:
            x.append((ix - Nx // 2) * dx)
            y.append((iy - Ny // 2) * dy)
            ix_arr.append(ix); iy_arr.append(iy)
        tries += 1
    return np.array(x), np.array(y), np.array(ix_arr), np.array(iy_arr)
```

**Angle, `'complex_gradient'` mode** (preferred — singularity-safe):

```python
def _angle_complex_gradient(E, dx, dy, ix, iy, k0, threshold):
    # ∂E/∂x, ∂E/∂y via central differences (numpy.gradient)
    dE_dy, dE_dx = np.gradient(E, dy, dx)  # numpy convention: axis 0 = y
    # Clamp denominator to avoid blowup at low-intensity / vortex cores
    E_clamp = np.where(np.abs(E) >= threshold * np.abs(E).max(),
                         E, threshold * np.abs(E).max())
    # k_⊥ = Im(∇E / E)
    kx_map = np.imag(dE_dx / E_clamp)
    ky_map = np.imag(dE_dy / E_clamp)
    # Sample at ray-origin pixels
    kx = kx_map[iy, ix]
    ky = ky_map[iy, ix]
    # Direction cosines
    L = kx / k0
    M = ky / k0
    sum_sq = L*L + M*M
    evanescent = sum_sq > 1.0
    N = np.where(evanescent, 0.0, np.sqrt(np.maximum(1.0 - sum_sq, 0.0)))
    return L, M, N, evanescent
```

**OPD initialization** so geometric ray-OPD accumulation continues
correctly from the wave-optical state:

```python
phi = np.angle(E)
opd_init = phi[iy, ix] / k0
```

### 4.5 Tests for Item 6

Add to `tests/test_raytrace/test_rays_from_field.py`:

```python
def test_plane_wave_normal_incidence():
    """A constant-amplitude, zero-phase field should produce rays
    all with (L=0, M=0, N=1)."""
    N = 128; dx = 2e-6; lam = 633e-9
    E = np.ones((N, N), dtype=complex)
    rays = rays_from_field(E, dx=dx, wavelength=lam, n_rays=100)
    assert np.allclose(rays.L, 0, atol=1e-6)
    assert np.allclose(rays.M, 0, atol=1e-6)
    assert np.allclose(rays.N, 1, atol=1e-6)
    assert np.all(rays.alive)

def test_tilted_plane_wave():
    """E = exp(i * kx * x) should give all rays L = kx / k0."""
    N = 256; dx = 1e-6; lam = 633e-9
    k0 = 2 * np.pi / lam
    theta = np.deg2rad(5.0)
    kx_expected = k0 * np.sin(theta)
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(1j * kx_expected * X)
    rays = rays_from_field(E, dx=dx, wavelength=lam, n_rays=200)
    assert np.allclose(rays.L, np.sin(theta), atol=1e-3)
    assert np.allclose(rays.M, 0, atol=1e-3)

def test_converging_spherical_wave_focuses():
    """A field with phase = -k0 r² / (2 f) should produce rays that
    converge to the focal point at z = f."""
    N = 256; dx = 1e-6; lam = 633e-9; f = 0.01
    k0 = 2 * np.pi / lam
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    R2 = X**2 + Y**2
    # Gaussian amplitude × converging spherical phase
    E = np.exp(-R2 / (50e-6)**2) * np.exp(-1j * k0 * R2 / (2 * f))
    rays = rays_from_field(E, dx=dx, wavelength=lam, n_rays=200,
                            placement='cdf')
    # Propagate each ray by L*z, M*z, N*z out to z = f and check
    # they converge to within a wavelength of the optical axis
    x_at_f = rays.x + (rays.L / rays.N) * f
    y_at_f = rays.y + (rays.M / rays.N) * f
    assert np.all(np.abs(x_at_f) < 5 * lam)
    assert np.all(np.abs(y_at_f) < 5 * lam)

def test_uniform_amplitude_uniform_density_cdf():
    """Uniform amplitude on a circular aperture should give roughly
    uniform 2D placement density (within bin-count tolerance)."""
    N = 256; dx = 1e-6
    R = 50e-6
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.where(X**2 + Y**2 <= R**2, 1.0 + 0j, 0.0)
    rays = rays_from_field(E, dx=dx, wavelength=633e-9,
                            n_rays=2000, placement='rejection',
                            random_state=42)
    # Check that placement is roughly uniform inside the aperture
    inside = rays.x**2 + rays.y**2 < R**2
    assert inside.sum() > 0.95 * len(rays.x)

def test_below_threshold_pixels_excluded():
    """Pixels below intensity_threshold should not be sampled."""
    N = 128; dx = 1e-6
    E = np.zeros((N, N), dtype=complex)
    E[60:68, 60:68] = 1.0  # tiny bright spot
    rays = rays_from_field(E, dx=dx, wavelength=633e-9,
                            n_rays=100, intensity_threshold=1e-4,
                            random_state=42)
    # All rays should fall inside the bright spot
    x_pix = np.round(rays.x / dx).astype(int) + N // 2
    y_pix = np.round(rays.y / dx).astype(int) + N // 2
    assert np.all((x_pix >= 60) & (x_pix < 68))
    assert np.all((y_pix >= 60) & (y_pix < 68))

def test_evanescent_rays_marked():
    """A field whose phase gradient exceeds k0 should produce
    evanescent-flagged rays."""
    N = 256; dx = 1e-6; lam = 633e-9
    k0 = 2 * np.pi / lam
    # phase gradient 2*k0 (definitely evanescent)
    x = (np.arange(N) - N // 2) * dx
    X, _ = np.meshgrid(x, x)
    E = np.ones((N, N), dtype=complex) * np.exp(1j * 2 * k0 * X)
    rays = rays_from_field(E, dx=dx, wavelength=lam, n_rays=50)
    assert np.any(~rays.alive)
    assert np.any(rays.error_code == RAY_EVANESCENT)

def test_opd_initialized_from_phase():
    """OPD at ray origin should equal phase(E_origin) / k0."""
    N = 128; dx = 1e-6; lam = 633e-9
    k0 = 2 * np.pi / lam
    x = (np.arange(N) - N // 2) * dx
    X, _ = np.meshgrid(x, x)
    phi = 0.3 * k0 * X  # smooth linear phase
    E = np.exp(1j * phi)
    rays = rays_from_field(E, dx=dx, wavelength=lam, n_rays=100,
                            random_state=42)
    expected_opd = phi[np.argmin(np.abs(x - rays.x[0])),
                       np.argmin(np.abs(x - rays.y[0]))] / k0
    # Compare ray-by-ray: opd should be 0.3 * X at ray origin
    expected = 0.3 * rays.x
    assert np.allclose(rays.opd, expected, atol=lam / 100)
```

### 4.6 Example for Item 6

`examples/rays_from_pupil_field.py`:
```python
"""Bridge a coherent pupil field into a ray bundle and trace it.

Useful when you have a measured / simulated exit-pupil field and
want to overlay a geometric ray fan on a defocus stack.
"""
import numpy as np
import lumenairy as la

# Build a representative converging field: Gaussian × thin-lens phase
N, dx, lam, f = 512, 2e-6, 633e-9, 0.05
src = la.Source.gaussian(N=N, dx=dx, wavelength=lam, w0=200e-6)
E_pupil = la.apply_thin_lens(src.E, f=f, wavelength=lam, dx=dx)

# Sample 80 rays from the pupil
rays = la.rays_from_field(E_pupil, dx=dx, wavelength=lam,
                           n_rays=80, placement='cdf')

print(f"Sampled {rays.n_rays} rays, "
      f"{rays.alive.sum()} alive")
print(f"Median tilt: L = {np.median(rays.L):.4f}, "
      f"M = {np.median(rays.M):.4f}")

# Project rays out to z = f and confirm they converge
xf = rays.x + (rays.L / rays.N) * f
yf = rays.y + (rays.M / rays.N) * f
print(f"RMS spot at focus: "
      f"{np.sqrt(np.mean(xf**2 + yf**2)) * 1e6:.2f} um")
```

### 4.7 Acceptance criteria for Item 6

- All tests in §4.5 pass.
- Function returns a `RayBundle` with all standard fields populated correctly.
- `'cdf'`, `'rejection'`, `'uniform'` placement modes all work and produce visibly different distributions for a non-separable test case (e.g. a vortex beam).
- `'complex_gradient'` angle method correctly handles a field with a phase singularity (vortex) without producing NaN.
- Evanescent rays are flagged with `RAY_EVANESCENT` and `alive=False`.
- Symbol exported from `lumenairy/__init__.py` and `lumenairy/raytrace/__init__.py`.
- Example `examples/rays_from_pupil_field.py` runs end-to-end.
- No existing test breaks.

---

## 5. Item 3 — `suggest_grid` heuristic (ALREADY IMPLEMENTED — DO NOT REIMPLEMENT)

This feature already exists as [`recommend_grid_for_prescription`](lumenairy/elements/lenses.py) at `lumenairy/elements/lenses.py:445`. It is more capable than the naive "fit a Gaussian to the source" heuristic the lens-designer repo lacked entirely:

- Sizes `(N, dx)` from the largest prescription `semi_diameter`.
- Optionally extends grid extent for DOE diffraction-order spread (`doe_orders_max`, `doe_period`, `doe_to_destination_distance`).
- Bounds `dx` by `wavelength / samples_per_wavelength` (default 4×, comfortable headroom above Nyquist 2×).
- Bounds `dx` further by `source_waist / samples_per_source_waist` (default 6×) if a source waist is supplied.
- Rounds `N` up to a power of two for FFT efficiency.
- Reports `dx_constraints` and `dx_limiting_constraint` so users understand which condition bound the recommendation.
- Sister function `check_grid_vs_apertures` for after-the-fact validation.

**Action items for the agent**: none. **Do not** add a competing `suggest_grid` function. Optionally, after Item 2 ships, add a thin **convenience wrapper** that takes a `CompositeOperator` (from Item 2) instead of a prescription dict — extracting the implicit apertures and powers from the operator chain — but only if it's clearly useful and only as a separate small PR. Discuss before implementing.

---

## 6. Rollout plan

1. **PR 1: Item 6 — `rays_from_field`** (low risk, additive single function). ~200 LOC + tests + 1 example. Target ~half a day.
2. **PR 2: Item 2 — operator algebra core**. Base class, primitives, composition, application delegation, `from_prescription`. ~600–800 LOC + tests + 2 examples. Target 1.5–2 days.
3. **PR 3 (optional, future): Item 2 Phase 2 — symbolic reduction**. Pattern-matching collapse rules for canonical operator chains (FreeSpace+ThinLens+FreeSpace → FourierTransform, etc.). Defer to a follow-up roadmap item; do **not** include in PR 2.

Each PR should:
- Update `CHANGELOG.md` with a `### Added` entry.
- Add a brief mention in `README.md` if user-visible.
- Pass `pytest tests/` end-to-end.
- Not touch existing public APIs.

---

## 7. References

- Nazarathy, M. & Shamir, J. "Fourier optics described by operator algebra." *JOSA* 70 (2), 150–159 (1980). The foundational paper for Item 2.
- Existing LumenAiry ABCD ground truth: `system_abcd`, `lens_abcd`, `surfaces_from_prescription` in [`raytrace/core.py`](lumenairy/raytrace/core.py).
- Existing LumenAiry grid sizing: `recommend_grid_for_prescription`, `check_grid_vs_apertures` in [`elements/lenses.py`](lumenairy/elements/lenses.py).
- Existing LumenAiry bundle types: `RayBundle` ([`raytrace/core.py:65`](lumenairy/raytrace/core.py)), `PathBundle` ([`propagators/hfpi.py`](lumenairy/propagators/hfpi.py)), `BeamletBundle` ([`propagators/gbd.py`](lumenairy/propagators/gbd.py)). Inter-bundle converters at [`raytrace/bundles.py`](lumenairy/raytrace/bundles.py).
- Reference implementation of operator algebra (paraxial-only, closure-based — do not copy verbatim, the architecture is different): `Neurophos/lens-designer/operators.py` lines 286–940 (`Operator`, `V`, `FT`, `Qu`, `Rprop`, `Aperture`).
- Reference implementation of rays-from-field (also paraxial-only, source repo — adapt the math, not the structure): `Neurophos/lens-designer/fields.py:186-307` (`generate_smart_rays`).

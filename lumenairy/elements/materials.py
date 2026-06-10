"""Tabulated dispersive materials for the grating-solver families (device-
geometry roadmap item 9, 2026-06-10).

Every project re-writes the same ``np.interp`` n/k loaders (the device port
this roadmap came from wrote four).  :class:`Material` is that loader, once:
a callable ``wl -> eps`` (the dispersive-material convention every solver
sweep in the library accepts: ``RCWAStack.add_layer``, ``PMMStack`` segments,
``PMM2DStack`` slots, the ``*_vs_wavelength`` entry points, and the
``materials={...}`` dict of ``SegmentStackGeometry`` / ``PMMStack.prepare``).

Conventions: tables hold ``n``/``k`` (refractive index, PUBLIC ``k >= 0``
lossy) against wavelength; ``eps = (n + i k)**2``.  Interpolation is linear
in ``n`` and ``k`` (not in eps); calls outside the tabulated range raise (no
silent extrapolation).
"""
from __future__ import annotations

import numpy as np


class Material:
    """A tabulated ``wl -> eps`` callable.

    >>> cu = Material.from_csv("McPeak_Cu.csv", wl_unit=1e-6)   # um column
    >>> stack.add_layer(70e-9, eps_cell=lambda wl: ...)         # or directly:
    >>> stack.add_layer(70e-9, eps=cu)                          # dispersive
    """

    def __init__(self, wavelengths, n, k=None, *, name=None):
        wl = np.asarray(wavelengths, dtype=float)
        n = np.asarray(n, dtype=float)
        k = (np.zeros_like(n) if k is None else np.asarray(k, dtype=float))
        if wl.ndim != 1 or wl.size < 2:
            raise ValueError("Material: need >= 2 tabulated wavelengths.")
        if not (np.all(np.isfinite(wl)) and np.all(np.isfinite(n))
                and np.all(np.isfinite(k))):
            raise ValueError("Material: table contains non-finite entries.")
        if n.shape != wl.shape or k.shape != wl.shape:
            raise ValueError("Material: n/k must match the wavelength grid.")
        order = np.argsort(wl)
        self.wl = wl[order]
        self.n = n[order]
        self.k = k[order]
        if np.any(np.diff(self.wl) <= 0):
            raise ValueError("Material: duplicate wavelengths in the table.")
        self.name = name

    def __call__(self, wl):
        w = float(wl)
        if w < self.wl[0] - 1e-15 or w > self.wl[-1] + 1e-15:
            raise ValueError(
                f"Material{('[' + self.name + ']') if self.name else ''}: "
                f"wavelength {w:.4g} m outside the tabulated range "
                f"[{self.wl[0]:.4g}, {self.wl[-1]:.4g}] m (no silent "
                f"extrapolation).")
        nv = float(np.interp(w, self.wl, self.n))
        kv = float(np.interp(w, self.wl, self.k))
        return complex((nv + 1j * kv) ** 2)

    def index(self, wl):
        """The complex refractive index ``n + i k`` at ``wl`` (PUBLIC
        ``k >= 0`` lossy) -- for the ``n_substrate``/``n_superstrate`` slots."""
        w = float(wl)
        self(w)                                   # range check
        return complex(np.interp(w, self.wl, self.n)
                       + 1j * np.interp(w, self.wl, self.k))

    @classmethod
    def from_csv(cls, path, *, wl_unit=1.0, name=None, delimiter=None,
                 skip_header=0):
        """Load a ``wavelength, n[, k]`` CSV/whitespace table.

        ``wl_unit`` scales the wavelength column to metres (``1e-6`` for um,
        ``1e-9`` for nm tables).  Lines that fail to parse as numbers are
        skipped (headers/comments), unless ``skip_header`` rows are given
        explicitly."""
        rows = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < int(skip_header):
                    continue
                parts = (line.replace(",", " ").split() if delimiter is None
                         else line.strip().split(delimiter))
                try:
                    vals = [float(x) for x in parts if x != ""]
                except ValueError:
                    continue
                if len(vals) >= 2:
                    rows.append(vals[:3])
        if len(rows) < 2:
            raise ValueError(
                f"Material.from_csv: no parsable 'wavelength, n[, k]' rows "
                f"in {path!r}.")
        arr = np.array([r + [0.0] * (3 - len(r)) for r in rows], dtype=float)
        return cls(arr[:, 0] * float(wl_unit), arr[:, 1], arr[:, 2],
                   name=name or str(path))

    def __repr__(self):
        return (f"Material({self.name or 'unnamed'}: "
                f"{self.wl[0]:.3g}..{self.wl[-1]:.3g} m, "
                f"{self.wl.size} points)")


__all__ = ["Material"]

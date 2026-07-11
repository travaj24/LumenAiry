"""Solver-independent geometry algebra for segmented z-stacks (device-geometry
roadmap item 2, 2026-06-10).

The most error-prone application code in a coated-grating model is GEOMETRY:
a conformal film following tooth tops + sidewalls + gap floors, a liner on
specific material-material interfaces only, a fill up to a cap plane.  Three
review rounds of a real device port were spent on exactly this.  It is pure
interval arithmetic on a z-stack of 1-D segment layers, independent of the
solver consuming it -- so it lives here, once, and feeds the PMM solver, the
RCWA cross-check, and the geometry viewer from ONE object (the
single-source-of-truth drift-proofing the application had to hand-build).

Representation: an ordered list (top -> bottom) of z-bands; each band is
``(thickness, [(x_lo, x_hi, material_key), ...])`` with x in ABSOLUTE metres
over one period and material keys as strings (eps resolved only at
``to_pmm_stack`` / ``to_rcwa_stack`` time).  ``BACKGROUND`` marks not-yet-
filled space.  All operations are exact interval arithmetic (band splitting
at new interface depths; wrap-aware in x).

The conformal ``coat`` is the L-infinity (square structuring element)
morphological dilation of the current SOLID set, minus the solid -- exact for
axis-aligned staircase cross-sections (square outer corners, the same
convention as a hand-built rectangle model), and it covers tops, sidewalls,
and exposed floors in one operation.
"""
from __future__ import annotations

import warnings

import numpy as np

BACKGROUND = "__background__"


def _norm_intervals(ivs, period, tol=1e-15):
    """Sort, clip, and merge same-material touching intervals."""
    out = []
    for lo, hi, m in sorted((max(0.0, lo), min(period, hi), m)
                            for lo, hi, m in ivs):
        if hi - lo <= tol:
            continue
        if out and out[-1][2] == m and lo - out[-1][1] <= tol:
            out[-1] = (out[-1][0], max(out[-1][1], hi), m)
        else:
            out.append((lo, hi, m))
    return out


def _complement(ivs, period, tol=1e-15):
    """Intervals of [0, period] not covered by ``ivs`` -- accepts (lo, hi)
    pairs or (lo, hi, mat) triples (assumed sorted/merged)."""
    out = []
    x = 0.0
    for iv in ivs:
        lo, hi = iv[0], iv[1]
        if lo - x > tol:
            out.append((x, lo))
        x = max(x, hi)
    if period - x > tol:
        out.append((x, period))
    return out


def _union(ivs, period, tol=1e-15):
    """Union of (lo, hi) intervals (material-blind), merged."""
    out = []
    for lo, hi in sorted(ivs):
        lo, hi = max(0.0, lo), min(period, hi)
        if hi - lo <= tol:
            continue
        if out and lo - out[-1][1] <= tol:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _dilate_x(ivs, t, period):
    """Expand each (lo, hi) by t on both sides, wrap-aware, then re-union."""
    grown = []
    for lo, hi in ivs:
        a, b = lo - t, hi + t
        if b - a >= period:
            return [(0.0, period)]
        if a < 0.0:
            grown.append((a + period, period))
            a = 0.0
        if b > period:
            grown.append((0.0, b - period))
            b = period
        grown.append((a, b))
    return _union(grown, period)


class SegmentStackGeometry:
    """Builder for a segmented (x, z) cross-section with geometry operations
    (ridges, conformal coats, interface liners, fills) that feeds every
    solver family from one object.

    >>> g = SegmentStackGeometry(period=550e-9)
    >>> g.add_ridges(350e-9, ridges=[(130e-9, 110e-9, 130e-9, "Cu"),
    ...                              (400e-9, 200e-9, 220e-9, "Cu")],
    ...              n_slices=4)
    >>> g.coat(6e-9, "Al2O3")            # tops + walls + floors, conformal
    >>> g.fill("LC")                     # remaining space
    >>> st = g.to_pmm_stack(materials={"Cu": eps_cu, "Al2O3": 3.1,
    ...                                "LC": lc_tensor},
    ...                     n_substrate=1.5, degree=12)
    """

    def __init__(self, period):
        self.period = float(period)
        self._bands = []                 # (thickness, [(lo, hi, mat), ...])

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #

    def add_band(self, thickness, segments):
        """Append one explicit band BELOW the current stack.  ``segments`` is
        ``[(width_fraction, material_key), ...]`` (widths sum to 1) in the
        solver segment convention; use :data:`BACKGROUND` (or ``None``) for
        not-yet-filled space."""
        t = float(thickness)
        if t <= 0:
            raise ValueError("add_band: thickness must be > 0.")
        ivs = []
        x = 0.0
        for w, m in segments:
            ivs.append((x, x + float(w) * self.period,
                        BACKGROUND if m is None else str(m)))
            x += float(w) * self.period
        if abs(x - self.period) > 1e-9 * self.period:
            raise ValueError("add_band: segment widths must sum to 1.")
        self._bands.append((t, _norm_intervals(ivs, self.period)))
        return self

    def add_ridges(self, thickness, *, ridges, n_slices=8):
        """Append a multi-ridge tapered region (item 1's builder, geometry
        side): ``ridges`` = ``(center, w_top, w_bottom, material_key)`` in
        ABSOLUTE metres, each tapering linearly about its own fixed center;
        the remainder of every slice is :data:`BACKGROUND` (fill it later).
        Wrap-aware; overlapping ridges raise."""
        n = int(n_slices)
        if n < 1:
            raise ValueError("add_ridges: n_slices must be >= 1.")
        rid = [(float(c), float(wt), float(wb), str(m))
               for c, wt, wb, m in ridges]
        dz = float(thickness) / n
        for k in range(n):
            zeta = (k + 0.5) / n
            ivs = []
            for c, wt, wb, m in rid:
                w = wt + (wb - wt) * zeta
                if w <= 0:
                    continue
                lo = (c - 0.5 * w) % self.period
                hi = lo + w
                if hi <= self.period:
                    ivs.append((lo, hi, m))
                else:
                    ivs.append((lo, self.period, m))
                    ivs.append((0.0, hi - self.period, m))
            ivs = _norm_intervals(ivs, self.period)
            for (l1, h1, _m1), (l2, _h2, _m2) in zip(ivs, ivs[1:]):
                if l2 < h1 - 1e-12 * self.period:
                    raise ValueError("add_ridges: ridges overlap.")
            for lo, hi in _complement(ivs, self.period):
                ivs.append((lo, hi, BACKGROUND))
            self._bands.append((dz, _norm_intervals(ivs, self.period)))
        return self

    # ------------------------------------------------------------------ #
    # geometry operations
    # ------------------------------------------------------------------ #

    def _solid_x(self, band_ivs):
        return _union([(lo, hi) for lo, hi, m in band_ivs
                       if m != BACKGROUND], self.period)

    def _split_z(self, planes):
        """Split bands at the given absolute depths (0 = top)."""
        planes = sorted({p for p in planes if p > 1e-15})
        out = []
        z = 0.0
        for t, ivs in self._bands:
            cuts = [p - z for p in planes if z + 1e-15 < p < z + t - 1e-15]
            edges = [0.0] + cuts + [t]
            for a, b in zip(edges, edges[1:]):
                out.append((b - a, list(ivs)))
            z += t
        self._bands = out

    def coat(self, t, mat, *, where="all"):
        """Grow a CONFORMAL film of thickness ``t`` over every exposed solid
        surface -- tooth tops, sidewalls, and gap floors in one operation --
        plus the bottom plane of the band stack (the substrate floor).  The
        film occupies previously-:data:`BACKGROUND` space (the L-infinity
        dilation of the solid set, minus the solid), so earlier materials are
        never overwritten; a new band of thickness ``t`` is added on top for
        the film over the topmost surfaces.

        ``where='all'`` (the conformal film) is the validated v1 contract;
        the tops/walls/floor SUBSET filters are deliberately not offered yet
        (a wrong subset classification is the exact failure class this layer
        exists to remove)."""
        t = float(t)
        if t <= 0:
            raise ValueError("coat: thickness must be > 0.")
        if where != "all":
            raise NotImplementedError(
                "coat: v1 implements the full conformal film (where='all'); "
                "subset filters (tops/walls/floor) are a follow-up.")
        mat = str(mat)
        if not self._bands:
            raise ValueError("coat: build geometry first.")
        # new empty band on top (film over the topmost solid), then split all
        # bands at every boundary +/- t so the dilation is exact per band
        self._bands.insert(0, (t, [(0.0, self.period, BACKGROUND)]))
        depths = [0.0]
        z = 0.0
        for tt, _ivs in self._bands:
            z += tt
            depths.append(z)
        total = z
        self._split_z([d + s * t for d in depths for s in (-1.0, 1.0)
                       if 0.0 < d + s * t < total])
        # per refined band: dilated solid = union over bands within the +/- t
        # z-window of their x-solids, each dilated by t in x
        zs = []
        z = 0.0
        for tt, ivs in self._bands:
            zs.append((z, z + tt, ivs))
            z += tt
        total = z
        new_bands = []
        for lo_z, hi_z, ivs in zs:
            window = []
            for lo2, hi2, ivs2 in zs:
                if lo2 < hi_z + t - 1e-15 and hi2 > lo_z - t + 1e-15:
                    window.extend(self._solid_x(ivs2))
            dil = _dilate_x(_union(window, self.period), t, self.period)
            # the substrate floor: within t of the stack bottom, the whole
            # exposed width is coated (film on the floor of every gap)
            if hi_z > total - t + 1e-15:
                dil = [(0.0, self.period)]
            film = []
            for dlo, dhi in dil:
                for blo, bhi, m in ivs:
                    if m != BACKGROUND:
                        continue
                    a, b = max(dlo, blo), min(dhi, bhi)
                    if b - a > 1e-15:
                        film.append((a, b, mat))
            keep = [(lo, hi, m) for lo, hi, m in ivs if m != BACKGROUND]
            covered = _union([(lo, hi) for lo, hi, _m in keep + film],
                             self.period)
            bg = [(lo, hi, BACKGROUND)
                  for lo, hi in _complement(covered, self.period)]
            new_bands.append((hi_z - lo_z,
                              _norm_intervals(keep + film + bg, self.period)))
        self._bands = new_bands
        return self

    def line_interface(self, mat_a, mat_b, *, t, mat, side="a"):
        """Insert a liner of thickness ``t`` wherever ``mat_a`` touches
        ``mat_b`` (vertical walls AND horizontal interfaces), carved out of
        the ``side`` material ('a' or 'b') so the outer geometry is
        unchanged -- the 1 nm Ta-on-specific-interfaces operation.

        v5.17 (audit P3-39): when the carved-side band at a horizontal
        interface is THINNER than ``t`` the horizontal liner cannot be
        carved and is omitted there; a ``UserWarning`` is emitted
        (previously a silent no-op leaving an inconsistent partial
        liner)."""
        t = float(t)
        a, b = str(mat_a), str(mat_b)
        carve = a if side == "a" else b
        if side not in ("a", "b"):
            raise ValueError("line_interface: side must be 'a' or 'b'.")
        mat = str(mat)
        # vertical (x-) interfaces within each band
        new = []
        for tt, ivs in self._bands:
            ivs = list(ivs)
            inserts = []
            for (l1, h1, m1), (l2, h2, m2) in zip(ivs, ivs[1:]):
                pair = {m1, m2}
                if pair == {a, b} and abs(l2 - h1) < 1e-15:
                    if m1 == carve:
                        inserts.append((h1 - t, h1, mat))
                    else:
                        inserts.append((l2, l2 + t, mat))
            # wrap seam
            if len(ivs) > 1:
                lN, hN, mN = ivs[-1]
                l0, h0, m0 = ivs[0]
                if {mN, m0} == {a, b} and abs(hN - self.period) < 1e-15 \
                        and abs(l0) < 1e-15:
                    if mN == carve:
                        inserts.append((hN - t, hN, mat))
                    else:
                        inserts.append((l0, l0 + t, mat))
            out = []
            for lo, hi, m in ivs:
                pieces = [(lo, hi)]
                for ilo, ihi, _im in inserts:
                    nxt = []
                    for plo, phi in pieces:
                        cl, ch = max(plo, ilo), min(phi, ihi)
                        if ch - cl > 1e-15:
                            if cl - plo > 1e-15:
                                nxt.append((plo, cl))
                            if phi - ch > 1e-15:
                                nxt.append((ch, phi))
                        else:
                            nxt.append((plo, phi))
                    pieces = nxt
                out.extend((plo, phi, m) for plo, phi in pieces)
            out.extend(inserts)
            new.append((tt, _norm_intervals(out, self.period)))
        self._bands = new
        # horizontal (z-) interfaces between consecutive bands
        self._split_z(self._hz_split_planes(a, b, t, carve))
        zb = []
        z = 0.0
        for tt, ivs in self._bands:
            zb.append([z, z + tt, list(ivs)])
            z += tt
        warned_thin = set()
        for i in range(len(zb) - 1):
            _z0, z1, up = zb[i]
            _z2, _z3, dn = zb[i + 1]
            for ulo, uhi, um in up:
                for dlo, dhi, dm in dn:
                    if {um, dm} != {a, b}:
                        continue
                    olo, ohi = max(ulo, dlo), min(uhi, dhi)
                    if ohi - olo <= 1e-15:
                        continue
                    # v5.17 audit (P3-39): when the carved-side band is
                    # THINNER than the liner thickness t, no split plane
                    # was inserted and the exact-equality carve gate
                    # below can never fire -- previously a silent no-op
                    # that left an inconsistent partial liner (vertical
                    # walls lined, this horizontal interface not).
                    # Warn once per (band, side).
                    if um == carve and (z1 - zb[i][0]) < t - 1e-15:
                        if (i, 'up') not in warned_thin:
                            warned_thin.add((i, 'up'))
                            warnings.warn(
                                f"line_interface: the {carve!r} band above "
                                f"the horizontal {a!r}|{b!r} interface at "
                                f"z={z1:.6g} m is only "
                                f"{z1 - zb[i][0]:.6g} m thick -- thinner "
                                f"than the liner thickness t={t:.6g} m -- "
                                f"so the horizontal liner is OMITTED at "
                                f"this interface (vertical-wall liners "
                                f"are still inserted).  Reduce t or "
                                f"thicken the band.",
                                UserWarning, stacklevel=2)
                    if um == carve and abs((z1 - zb[i][0])) >= t - 1e-15:
                        # carve the bottom t of the upper band's overlap
                        if abs((z1 - zb[i][0]) - t) < 1e-15:
                            zb[i][2] = self._replace_x(zb[i][2], olo, ohi,
                                                       mat)
                    elif dm == carve and abs((zb[i + 1][1] - z1)) >= t - 1e-15:
                        if abs((zb[i + 1][1] - z1) - t) < 1e-15:
                            zb[i + 1][2] = self._replace_x(zb[i + 1][2], olo,
                                                           ohi, mat)
                    elif dm == carve and (zb[i + 1][1] - z1) < t - 1e-15:
                        if (i, 'dn') not in warned_thin:
                            warned_thin.add((i, 'dn'))
                            warnings.warn(
                                f"line_interface: the {carve!r} band below "
                                f"the horizontal {a!r}|{b!r} interface at "
                                f"z={z1:.6g} m is only "
                                f"{zb[i + 1][1] - z1:.6g} m thick -- "
                                f"thinner than the liner thickness "
                                f"t={t:.6g} m -- so the horizontal liner "
                                f"is OMITTED at this interface "
                                f"(vertical-wall liners are still "
                                f"inserted).  Reduce t or thicken the "
                                f"band.", UserWarning, stacklevel=2)
        self._bands = [(z1 - z0, _norm_intervals(ivs, self.period))
                       for z0, z1, ivs in zb]
        return self

    def _hz_split_planes(self, a, b, t, carve):
        """Depths at which the z-liner needs band splits: t inside the carved
        side of every horizontal a|b interface."""
        planes = []
        z = 0.0
        bands = self._bands
        for i in range(len(bands) - 1):
            z1 = z + bands[i][0]
            up, dn = bands[i][1], bands[i + 1][1]
            touch = any({um, dm} == {a, b}
                        and min(uhi, dhi) - max(ulo, dlo) > 1e-15
                        for ulo, uhi, um in up for dlo, dhi, dm in dn)
            if touch:
                for ulo, uhi, um in up:
                    if um == carve:
                        planes.append(z1 - t)
                for dlo, dhi, dm in dn:
                    if dm == carve:
                        planes.append(z1 + t)
            z = z1
        return planes

    @staticmethod
    def _replace_x(ivs, lo, hi, mat):
        out = []
        for plo, phi, m in ivs:
            cl, ch = max(plo, lo), min(phi, hi)
            if ch - cl > 1e-15:
                if cl - plo > 1e-15:
                    out.append((plo, cl, m))
                out.append((cl, ch, mat))
                if phi - ch > 1e-15:
                    out.append((ch, phi, m))
            else:
                out.append((plo, phi, m))
        return out

    def fill(self, mat):
        """Replace every remaining :data:`BACKGROUND` region with ``mat``."""
        mat = str(mat)
        self._bands = [
            (t, _norm_intervals([(lo, hi, mat if m == BACKGROUND else m)
                                 for lo, hi, m in ivs], self.period))
            for t, ivs in self._bands]
        return self

    # ------------------------------------------------------------------ #
    # consumers
    # ------------------------------------------------------------------ #

    def layers(self):
        """The cross-section as ``[(thickness, [(width_fraction, key), ...])]``
        (top -> bottom) -- the solver segment convention."""
        out = []
        for t, ivs in self._bands:
            if any(m == BACKGROUND for _l, _h, m in ivs):
                raise ValueError(
                    "SegmentStackGeometry: BACKGROUND regions remain; call "
                    "fill(...) (or assign them explicitly) before exporting.")
            out.append((t, [((hi - lo) / self.period, m)
                            for lo, hi, m in ivs]))
        return out

    def to_pmm_stack(self, *, materials=None, **stack_kwargs):
        """Build a :class:`~lumenairy.elements.pmm.PMMStack` from this
        geometry.  ``materials`` maps key -> eps (scalar / (3,3) / wl-callable);
        omitted keys stay as MATERIAL KEYS for ``stack.prepare()`` swapping.

        The material KEY NAMES ride along (application feedback 2026-06-10):
        ``stack.plot_geometry()`` legends read ``"LC"``/``"Ta"`` instead of
        eps values, and ``layer_absorption(by_material=True)`` attributes by
        KEY -- so twin keys mapped to the SAME eps (the ``"Cu"``/``"CuCol"``
        under-tooth-liner trick) split the loss map as they split the
        geometry."""
        from .pmm import PMMStack
        materials = dict(materials or {})
        st = PMMStack(self.period, **stack_kwargs)
        names = {}
        labels = []
        for t, segs in self.layers():
            st.add_layer(t, segments=[(w, materials.get(m, m))
                                      for w, m in segs])
            labels.append([m for _w, m in segs])
            for _w, m in segs:
                v = materials.get(m, m)
                if isinstance(v, str) or callable(v):
                    names[v if isinstance(v, str) else m] = m
                else:
                    M = np.asarray(v)
                    disp = complex(M[0, 0]) if M.ndim == 2 else complex(M)
                    names.setdefault(disp, m)
        st._material_names = names
        st._segment_labels = labels
        return st

    def to_rcwa_stack(self, *, materials, n_x=1024, **stack_kwargs):
        """Build a :class:`~lumenairy.elements.rcwa.RCWAStack` (1-D) from this
        geometry by exact-boundary pixelation at ``n_x`` samples per period
        (``materials`` must resolve every key; RCWA layers hold values, not
        keys).

        ANISOTROPIC materials are supported (application feedback ask 4,
        2026-06-10): a band containing any ``(3, 3)`` tensor material is
        pixelated into an ``eps_tensor_cell`` with scalar materials promoted
        to ``eps * I3``; scalar-only bands keep the ``eps_cell`` path."""
        from .rcwa import RCWAStack
        st = RCWAStack(self.period, **stack_kwargs)
        nx = int(n_x)
        xs = (np.arange(nx) + 0.5) / nx * self.period
        eye3 = np.eye(3, dtype=complex)
        for t, ivs in self._bands:
            vals = []
            for lo, hi, m in ivs:
                if m == BACKGROUND:
                    raise ValueError(
                        "to_rcwa_stack: BACKGROUND remains; fill(...) first.")
                v = np.asarray(materials[m], dtype=complex)
                if v.ndim not in (0, 2) or (v.ndim == 2
                                            and v.shape != (3, 3)):
                    raise ValueError(
                        f"to_rcwa_stack: material {m!r} must be a scalar or "
                        f"a (3, 3) tensor, got shape {v.shape}.")
                vals.append((lo, hi, v))
            if any(v.ndim == 2 for _lo, _hi, v in vals):
                cell = np.empty((nx, 1, 3, 3), dtype=complex)
                cell[:] = np.nan
                for lo, hi, v in vals:
                    M3 = v if v.ndim == 2 else complex(v) * eye3
                    cell[(xs >= lo) & (xs < hi), 0] = M3
                # BSDF-nit (AUDIT_BSDF_SEGMENT_GEOMETRY): fail loud if the
                # intervals didn't tile [0, period) -- a residual NaN eps
                # (from the NaN seed) would otherwise flow silently into the
                # RCWA layer.
                if np.any(np.isnan(cell)):
                    _ng = int(np.isnan(cell[:, 0, 0, 0]).sum())
                    raise ValueError(
                        f"to_rcwa_stack: interval set does not tile "
                        f"[0, period); {_ng} pixel column(s) left unassigned "
                        f"(NaN eps).  Ensure the layer's intervals cover the "
                        f"full period without gaps.")
                st.add_layer(t, eps_tensor_cell=cell)
            else:
                col = np.empty(nx, dtype=complex)
                col[:] = np.nan
                for lo, hi, v in vals:
                    col[(xs >= lo) & (xs < hi)] = complex(v)
                if np.any(np.isnan(col)):
                    _ng = int(np.isnan(col).sum())
                    raise ValueError(
                        f"to_rcwa_stack: interval set does not tile "
                        f"[0, period); {_ng} pixel(s) left unassigned (NaN "
                        f"eps).  Ensure the layer's intervals cover the full "
                        f"period without gaps.")
                st.add_layer(t, eps_cell=col[:, None])
        return st

    def plot(self, ax=None, material_colors=None):
        """Draw the cross-section exactly (one rectangle per interval) --
        "the picture IS the model"."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        if ax is None:
            _f, ax = plt.subplots(figsize=(6, 4))
        cmap = plt.get_cmap("tab10")
        colors = dict(material_colors or {})
        z = 0.0
        for t, ivs in self._bands:
            for lo, hi, m in ivs:
                if m not in colors:
                    colors[m] = ("0.95" if m == BACKGROUND
                                 else cmap(len(colors) % 10))
                ax.add_patch(Rectangle((lo, -z - t), hi - lo, t,
                                       facecolor=colors[m], edgecolor="none"))
            z += t
        handles = [__import__("matplotlib").patches.Rectangle(
            (0, 0), 1, 1, facecolor=c) for c in colors.values()]
        ax.legend(handles, list(colors), fontsize=7, loc="center left",
                  bbox_to_anchor=(1.0, 0.5))
        ax.set_xlim(0, self.period)
        ax.set_ylim(-z, 0)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m] (0 = top)")
        return ax


__all__ = ["SegmentStackGeometry", "BACKGROUND"]

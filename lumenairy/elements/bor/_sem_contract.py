"""The SEM manufactured-element contract for the BOR stack solver.

WHAT THE ENRICHMENT WINDOW IS, AND WHY IT MANUFACTURES ELEMENTS.
``BORStack._solve_sem`` does not mesh layer ``i`` on layer ``i``'s own ring
walls.  It meshes it on the **union of the walls of layers ``i-1``, ``i`` and
``i+1``**::

    u = set(walls[i])
    if i > 0:              u |= set(walls[i - 1])
    if i + 1 < len(walls): u |= set(walls[i + 1])

That union is deliberate and is what makes the mortar between neighbouring
layers nearly conforming, hence cheap and accurate.  It is also the whole of
this error class: two neighbouring layers whose walls differ by a small
``delta`` manufacture an element of width exactly ``delta`` in BOTH of their
meshes, in neither of which any single layer asked for it.  And because the
window reaches ``i-1`` AND ``i+1``, a WALL-FREE uniform spacer placed between
two ring layers inherits both of their wall sets and carries the sliver in a
mesh that has no walls of its own -- a spacer is not a refuge.

WHAT A MANUFACTURED ELEMENT DOES.  The spectral-element Jacobian makes the
nodal stiffness scale as ``1 / w^2``, so the layer's modal spectrum acquires
AXIAL WAVENUMBERS far above any value the medium can physically support, the
mode match at the interface conditions on them, and the per-order answer moves.
Measured on the scoping's ladder (``delta/Rbig`` from 1e-1 to 1e-7, Rbig = 24,
k0 = 2, m = 1, degrees 6 / 8 / 12;
``docs/audits/SCOPE_BOR_MULTILAYER_GUARDS_2026_09_12.md`` section 4.2): at
``delta/Rbig = 1e-7`` the per-order ``R`` has moved **586x (degree 12) to
5,428x (degree 8)** the physical wall shift the ``delta`` represents, against
the 1-D guard's attribution bar of 100x; ``|q|max / (n_max k0)`` reads
**5.014e+06**; the interface ``rcond`` is 6.7e-10; and **zero** warnings are
emitted anywhere on the ladder.  A control that keeps the same geometry but
places the two walls THREE layers apart -- so the ``+-1`` window never spans
both wall sets -- is clean on every quantity (narrowest element 1.500 against
2.400e-06, ``|q|max``/ceiling 27.22 against 5.014e+06, closure 3.550e-10
against 2.372e-04).  The geometry is harmless; the window's union of it is not.

WHAT THIS CONTRACT KEYS ON, AND WHAT IT REFUSES TO KEY ON.  A CONJUNCTION of
a geometric cause and a spectral consequence, and NEITHER conjunct is the
energy violation:

  (a) the POST-window, POST-DPW, POST-``equalize_meshes`` breakpoint set of a
      layer contains a cell narrower than ``_BOR_MIN_ELEM_FRAC * Rbig`` whose
      two ENCLOSING WALLS come from DIFFERENT layers -- the own-scale test: the
      union manufactured it, no single layer asked for it;  **and**
  (b) that layer's modal spectrum reads
      ``|q|max / (n_max k0) > _BOR_Q_EXCESS`` -- a mode called propagating that
      no propagating mode of this medium can be.

THE ENERGY IS NOT USABLE HERE, and that is measured rather than assumed.  Over
the same ladder the closure's spread across three OpenBLAS kernels is
**70.90x**, straddling the 1-D ``_SLIVER_TRIGGER_BAR`` of 1e-3 -- the same
solve would be arbitrated on one kernel and pass silently on another.
Independently the damage is ENERGY-INVISIBLE over four decades: at
``delta/Rbig = 1e-04`` the closure sits at its healthy 1.04e-09 baseline while
the per-order answer has already moved 4.00e-04.  A ported super-unity screen
at the 1-D bars fires on 4 of 60 union rungs and 2 of 40 within-layer rungs --
the catastrophic tail only.  By contrast the geometric conjunct is
kernel-EXACT (spread 1.0000x) and the spectral one kernel-stable (1.1895x).

WHAT IS DELIBERATELY OUT OF THE CONTRACT.  A narrow element the USER asked for
-- an annular liner inside ONE layer's own segment list -- is not manufactured
by the library and is never REFUSED.  It is warned about when it drives the
spectrum past the same screen, because it is the same damage and the caller
should hear about it, but the caller prescribed the geometry and the library
does not overrule that.  ``basis='fd'`` is structurally immune (one uniform
radial grid that does not know where the walls are) and no contract is applied
to it.
"""
from __future__ import annotations

import warnings

import numpy as np

#: FAIL-BEFORE SWITCH (5.45.1).  ``False`` restores the pre-fix behaviour bit
#: for bit: no refusal, no warning, and the wrong number returned.  A switch,
#: not a policy -- the contract changes nothing on any solve that does not trip
#: both conjuncts.
BOR_SEM_MESH_GUARD = True

#: CONJUNCT (b), THE DECIDING QUANTITY: ``|q|max / (n_max k0)``, the layer's
#: largest axial wavenumber against the largest one its own medium can support.
#: A pure ratio of the discretisation to the physics, and the only candidate
#: with a two-sided gap that survives the kernel matrix.
#:
#: MEASURED POPULATIONS.  The false-positive census was WIDENED before this bar
#: was accepted -- 86 ordinary geometry families over degrees 6/8/12/16 plus a
#: taper staircase out to 256 slices
#: (``validation/probe_fix_bor_guards/s4_sem_census.py`` and
#: ``s4_deep_taper.py``), against the scoping's four families at 64 slices --
#: and it is re-measured on the running build by
#: ``test_fix_bor_multilayer_guards.py::test_ordinary_geometry_census``:
#:
#:   ORDINARY, 72 non-taper families, degrees 6-16               14.8 .. 164.4
#:   ORDINARY, separated control, every rung, degrees 6/8/12     16.06 .. 47.16
#:   ORDINARY, taper staircase, 4 .. 256 slices, degrees 8/12    22.1 .. 942.0
#:   ORDINARY, taper staircase at 256 slices, k0 = 0.8, deg 8   **1134.2**
#:   ---------------------------------------------------------  --------------
#:   DAMAGING, the mildest REFUSED rung of the union ladder      1.5934e+05
#:     (delta/Rbig = 1e-6, degree 6)
#:   DAMAGING, the union ladder at delta/Rbig = 1e-7             1.5933e+06 ..
#:                                                               5.0153e+06
#:   DAMAGING, the c5 onset over five (Rbig, k0) cases           2.152e+05 ..
#:                                                               1.003e+07
#:
#: 1e4 sits **0.95 decades (8.82x) above the worst ordinary geometry measured**
#: and **1.20 decades (15.9x) below the mildest rung it must refuse** --
#: two-sided, with about a decade on each side.  Kernel spread of the quantity
#: itself: 1.1895x.
#:
#: The scoping quoted 1.63 decades of headroom on the ordinary side; that was a
#: SAMPLE property of a census that stopped at a 64-slice taper (235.8).  The
#: widened census reaches 1134.2, and the honest margin is 0.95 decades.  A
#: LOWER k0 RAISES this ratio, so the low-k0 taper arm is the demanding one and
#: is the reason the census sweeps k0 as well as the slice count.
_BOR_Q_EXCESS = 1.0e+04

#: CONJUNCT (a), THE ATTRIBUTION: a cell narrower than this fraction of
#: ``Rbig``, between two walls from DIFFERENT layers.
#:
#: HONESTLY A FACTOR-30 QUANTITY, and it attributes rather than decides for
#: exactly that reason.  Scoping section 4.7 swept ``k0`` at fixed ``Rbig``
#: (moving the local wavelength 16x while ``Rbig`` stood still) and ``Rbig`` at
#: fixed ``k0``, and found that NO width scale holds still: over five
#: ``(Rbig, k0)`` cases spanning ``Rbig/lambda`` from 4.7 to 74.9 the onset
#: spreads by 30.0x in ``w/Rbig``, 60.0x in ``w/lambda`` and 45.0x in
#: ``w/h_ordinary``.  ``w/Rbig`` is the tightest but only by 1.5x to 2x.  1e-6
#: is the CONSERVATIVE (widest) end of the measured onset band
#: (1.000e-07 .. 3.000e-06), so the geometric conjunct fires at or before the
#: onset on every case measured.  Kernel spread: 1.0000x -- exact.
_BOR_MIN_ELEM_FRAC = 1.0e-6

#: THE DEGRADATION BAND: a manufactured cell between ``_BOR_MIN_ELEM_FRAC``
#: and this fraction of ``Rbig`` emits a ``UserWarning`` and NEVER a refusal.
#:
#: SET BY THE FALSE-POSITIVE CENSUS, NOT BY THE ACCURACY, and the census was
#: measured BEFORE the edge was chosen.  That order is the 2-D peer's round-4
#: correction: a census margin measured on four geometry families is a SAMPLE
#: property, not a library one.
#:
#: THE BINDING ORDINARY GEOMETRY is the taper staircase -- a cone sliced into
#: layers, each carrying its own ring radius, so ADJACENT slices' walls differ
#: by ``(r_top - r_bot) / n_slices``.  Its manufactured cell HALVES with every
#: doubling of the slice count while ``|q|max / ceiling`` DOUBLES.  Measured
#: (``validation/probe_fix_bor_guards/s4_sem_census.py``, ``s4_deep_taper.py``;
#: Rbig = 24, cone 8 -> 2 over height 1.2):
#:
#:     slices    4      8     16     32     64    128    256
#:     w/Rbig  3.1e-2 3.1e-2 1.6e-2 7.8e-3 3.9e-3 2.0e-3 **9.77e-4**
#:
#: **THE SCOPING'S CANDIDATE EDGE OF 1e-3 IS REFUTED BY THIS CENSUS**: a
#: 256-slice taper lands at 9.766e-04, i.e. 0.977x -- INSIDE the band it would
#: have warned on.  The scoping's own note said its 3.9x margin at 64 slices
#: was sample-scoped and had to be re-measured before the edge was fixed; this
#: is that measurement, and it moved the edge a decade.
#:
#: At 1e-4 the 256-slice taper carries **9.77x (0.99 decades)** and every one of
#: the 86 ordinary families measured lands outside the band.  The edge still
#: WARNS on a deep enough taper -- a ~2,500-slice one -- which is the intent and
#: the same surface the 2-D contract deliberately warns on; it can never REFUSE
#: one, because the refusal's geometric conjunct is two decades further down at
#: 1e-6 of Rbig, which a taper reaches only at ~6 million slices.
_BOR_SLIVER_BAND_FRAC = 1.0e-4

#: Position tolerance for matching a mesh breakpoint to a ring wall.  The mesh
#: builder itself merges breakpoints closer than ``1e-12 * Rbig``, so a wall
#: that survives into the mesh survives at this tolerance.
_BOR_WALL_ATOL_FRAC = 1.0e-12


class BORSemMeshError(ValueError):
    """Raised by ``BORStack.solve(basis='sem')`` when the ``+-1`` enrichment
    window has MANUFACTURED a radial element too narrow for the spectral-element
    basis to resolve, AND the affected layer's modal spectrum shows the
    spurious axial wavenumbers that proves it.

    A subclass of ``ValueError`` so existing ``except ValueError`` handlers are
    unaffected.  The message names the layer, the manufactured width, the two
    walls and which layers they came from, the measured spectral excess, and
    the remedy."""


def _wall_sources(walls):
    """``{position: {layer indices that asked for this wall}}``.

    Positions are compared as exact floats because that is how they travel:
    ``_solve_sem`` puts the CALLER's wall coordinates into a ``set`` and the
    mesh builder passes them through ``np.unique`` without arithmetic.  A wall
    two layers spell slightly differently is, correctly, two walls.
    """
    src = {}
    for i, wl in enumerate(walls):
        for w in wl:
            src.setdefault(float(w), set()).add(i)
    return src


def measure_layer(bnd, layer_index, walls, Rbig, q, n_max, k0):
    """MEASURE one layer's mesh against the contract.  Never raises, never
    warns -- the census and the guard read the same numbers from here, which is
    what makes the census a statement about the shipped behaviour rather than
    about a parallel re-implementation.

    ``bnd`` is the POST-window, POST-DPW, POST-``equalize_meshes`` breakpoint
    array; ``walls`` the per-layer wall lists; ``q`` the layer's modal spectrum;
    ``n_max`` the largest refractive index anywhere in the layer.

    Returns a dict with, per layer:

    ``w_min``            the narrowest element, absolute and as a fraction of
                         ``Rbig``
    ``w_min_union``      the narrowest element whose two ENCLOSING walls come
                         from DIFFERENT layers (``inf`` when none does)
    ``attribution``      which layers those two walls came from
    ``q_excess``         ``|q|max / (n_max k0)``
    """
    bnd = np.asarray(bnd, dtype=float)
    Rbig = float(Rbig)
    atol = _BOR_WALL_ATOL_FRAC * Rbig
    src = _wall_sources(walls)
    wpos = np.array(sorted(src), dtype=float) if src else np.zeros(0)

    def enclosing(x, side):
        """The nearest wall at or below (``side < 0``) / at or above
        (``side > 0``) ``x``; ``None`` when the domain end is nearer."""
        if wpos.size == 0:
            return None
        if side < 0:
            j = int(np.searchsorted(wpos, x + atol, side="right")) - 1
        else:
            j = int(np.searchsorted(wpos, x - atol, side="left"))
        if j < 0 or j >= wpos.size:
            return None
        return float(wpos[j])

    widths = np.diff(bnd) if bnd.size > 1 else np.zeros(0)
    w_min = float(np.min(widths)) if widths.size else float("inf")
    w_union = float("inf")
    attrib = None
    for j in range(widths.size):
        a, b = float(bnd[j]), float(bnd[j + 1])
        lo, hi = enclosing(a, -1), enclosing(b, +1)
        if lo is None or hi is None:
            continue
        slo, shi = src.get(lo, set()), src.get(hi, set())
        if not slo or not shi or not slo.isdisjoint(shi):
            continue                       # one layer asked for both -> own
        if widths[j] < w_union:
            w_union = float(widths[j])
            attrib = dict(r_lo=lo, r_hi=hi,
                          layers_lo=sorted(slo), layers_hi=sorted(shi))
    qa = np.asarray(q)
    den = float(np.real(n_max)) * float(np.real(k0))
    q_excess = (float(np.max(np.abs(qa))) / den
                if qa.size and den > 0.0 else float("nan"))
    return dict(layer=int(layer_index), Rbig=Rbig,
                w_min=w_min, w_min_frac=w_min / Rbig if Rbig > 0 else float("inf"),
                w_min_union=w_union,
                w_min_union_frac=(w_union / Rbig if (Rbig > 0
                                                     and np.isfinite(w_union))
                                  else float("inf")),
                attribution=attrib, q_excess=q_excess,
                n_elements=int(max(bnd.size - 1, 0)))


def verdict(rec):
    """``'ok' | 'warn_manufactured' | 'warn_own' | 'refuse'`` for one measured
    layer.  Pure function of the record, so a census can tabulate the same
    verdicts the solve would reach."""
    excess = rec["q_excess"]
    hot = np.isfinite(excess) and excess > _BOR_Q_EXCESS
    fu = rec["w_min_union_frac"]
    fa = rec["w_min_frac"]
    if fu < _BOR_MIN_ELEM_FRAC and hot:
        return "refuse"
    if fu < _BOR_SLIVER_BAND_FRAC:
        return "warn_manufactured"
    if fa < _BOR_MIN_ELEM_FRAC and hot:
        return "warn_own"
    return "ok"


def enforce(records):
    """Apply the contract to a solve's measured layers.

    No-op when :data:`BOR_SEM_MESH_GUARD` is ``False``.  Raises
    :class:`BORSemMeshError` on the first refusing layer; otherwise emits one
    ``UserWarning`` per warning layer and returns.
    """
    if not BOR_SEM_MESH_GUARD:
        return
    for rec in records:
        v = verdict(rec)
        if v == "refuse":
            at = rec["attribution"] or {}
            raise BORSemMeshError(
                "BORStack.solve(basis='sem'): layer %d's radial mesh contains "
                "a MANUFACTURED element of width %.6g (%.3e of Rbig = %.6g), "
                "between the wall at r = %.12g (asked for by layer(s) %s) and "
                "the wall at r = %.12g (asked for by layer(s) %s).  NO SINGLE "
                "LAYER ASKED FOR THAT CELL: the +-1 enrichment window unions "
                "each layer's walls with its neighbours', so two layers whose "
                "walls differ by a small delta manufacture a cell of exactly "
                "that delta in BOTH meshes (and a wall-free spacer BETWEEN "
                "two ring layers inherits both wall sets, so a spacer is not "
                "a refuge).  The spectral-element Jacobian makes the nodal "
                "stiffness scale as 1/w^2, and this layer's spectrum shows "
                "it: |q|max / (n_max k0) = %.4g against the %.0e bar -- an "
                "axial wavenumber %.4gx larger than any propagating mode of "
                "this medium can carry.  Measured on this family, an answer "
                "in this state has moved 586x to 5,428x the physical wall "
                "shift the delta represents (bar 100x) while its energy "
                "closure stayed at its healthy 1e-10 baseline, so nothing "
                "else would have told you.  REMEDIES: give the two layers the "
                "SAME wall (a coincidence of %.3e of Rbig is below the %.0e "
                "this mesh can resolve), or separate them so the +-1 window "
                "never "
                "spans both wall sets (TWO wall-free spacers, not one), or "
                "use basis='fd', which puts every layer on one uniform radial "
                "grid and is structurally immune.  To restore the previous "
                "behaviour and receive the number instead of this error, set "
                "lumenairy.elements.bor._sem_contract.BOR_SEM_MESH_GUARD = "
                "False."
                % (rec["layer"], rec["w_min_union"], rec["w_min_union_frac"],
                   rec["Rbig"], at.get("r_lo", float("nan")),
                   at.get("layers_lo"), at.get("r_hi", float("nan")),
                   at.get("layers_hi"), rec["q_excess"], _BOR_Q_EXCESS,
                   rec["q_excess"], rec["w_min_union_frac"],
                   _BOR_MIN_ELEM_FRAC))
    for rec in records:
        v = verdict(rec)
        if v == "warn_manufactured":
            at = rec["attribution"] or {}
            warnings.warn(
                "BORStack.solve(basis='sem'): layer %d's radial mesh contains "
                "a MANUFACTURED element of %.3e of Rbig (walls r = %.10g from "
                "layer(s) %s and r = %.10g from layer(s) %s), inside the "
                "degradation band [%.0e, %.0e).  No single layer asked for "
                "that cell -- the +-1 enrichment window unions each layer's "
                "walls with its neighbours'.  The answer is still returned; "
                "its per-order accuracy degrades as the cell narrows (|q|max "
                "/ (n_max k0) = %.4g here, ordinary geometry reads 16 .. 236) "
                "and below %.0e of Rbig it is refused.  Give the two layers "
                "the same wall, or use basis='fd'."
                % (rec["layer"], rec["w_min_union_frac"],
                   at.get("r_lo", float("nan")), at.get("layers_lo"),
                   at.get("r_hi", float("nan")), at.get("layers_hi"),
                   _BOR_MIN_ELEM_FRAC, _BOR_SLIVER_BAND_FRAC,
                   rec["q_excess"], _BOR_MIN_ELEM_FRAC),
                stacklevel=3)
        elif v == "warn_own":
            warnings.warn(
                "BORStack.solve(basis='sem'): layer %d's radial mesh contains "
                "an element of %.3e of Rbig that the LAYER'S OWN segment list "
                "asked for, and its spectrum reads |q|max / (n_max k0) = %.4g "
                "against the %.0e screen -- axial wavenumbers no propagating "
                "mode of this medium can carry.  This is NOT refused: you "
                "prescribed the geometry and the library does not overrule "
                "it.  But the spectral-element error stops converging in "
                "DEGREE at this width (measured: degree 16 becomes 144x WORSE "
                "than degree 6), and the energy closure does not show it.  "
                "Widen the feature, or model it as a permittivity average."
                % (rec["layer"], rec["w_min_frac"], rec["q_excess"],
                   _BOR_Q_EXCESS),
                stacklevel=3)

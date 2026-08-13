# Rigorous (RCWA) order table for the design-121 Dammann DOE.
#
# WHAT THIS REPLACES.  ``_d121_common.order_table`` returns the DFT of the
# Dammann cell's complex transmittance -- the THIN-ELEMENT (scalar) table.  It
# is angle-blind, polarization-blind, lossless by construction and knows
# nothing about the etched relief that realises the phase.  This module
# computes the same 32 order amplitudes by solving Maxwell's equations on the
# reconstructed surface-relief cell (2-D crossed-grating RCWA), over an angle x
# polarization grid, and caches the result.
#
# THE STRUCTURE IS RECONSTRUCTED, NOT READ.  Read ``STRUCTURE ASSUMPTIONS``
# below before using any number this module produces.  Nothing in the design
# tree records the DOE's material or its etch depth; both are DERIVED here from
# the phase design under assumptions that are stated, defaulted and
# PARAMETERISED, so a corrected value is one re-run (change ``n_doe`` -- or pass
# ``--n-doe`` on the CLI -- and every cached table re-keys automatically).
#
# LOCAL-ONLY?  No.  This module is import-safe and design-agnostic: it takes a
# phase-level map and a structure record.  Only :func:`design121_structure`
# touches the design-121 cell, and it can read the cached ``.npy`` that
# ``_d121_common.order_table`` writes without needing the ``.zmx`` at all.
#
# cp1252-safe ASCII only.
"""Rigorous RCWA order table for a multi-level surface-relief DOE.

STRUCTURE ASSUMPTIONS
=====================
Stated loudly, because the physical DOE is NOT recorded anywhere in the design
tree and every number this module produces is conditional on them.

A1. MATERIAL -- ASSUMED, NOT RECORDED.  Zemax surfaces 9 and 11 of
    ``20260707 dll Tx02-MSOP16.zmx`` are ``DGRATING`` surfaces with **no
    ``GLAS`` line and ``DISZ 0``**: in Zemax that is a zero-thickness element in
    air.  The design-study runner's ``_NEW_GLASSES`` table registers N-SK2,
    N-SF1, N-PK52A, N-LAK8 and N-LAK9 -- all refractive-group glasses, none at
    the DOE.  The only DOE substrate ever named in this project lineage is
    FUSED SILICA (``tx_design_study_sim`` registers an ``F_SILICA`` alias for
    the Design-36/71-era part), so that is the DESIGN-CONSISTENT assumption
    taken here:

        n_doe = 1.446804   (fused silica, Malitson, at lambda = 1.31 um)

    ``n_doe`` is a parameter of :class:`DoeStructure` and enters the cache key.
    A corrected material is one re-run.

A2. RELIEF DEPTH -- DERIVED, NOT RECORDED.  No etch depth exists in the tree.
    The 8 phase levels (``DAMMANN_PHASELEVELS = 8``, confirmed: the cached cell
    takes exactly 8 distinct values, multiples of 2*pi/8) are realised as an
    8-level staircase of vertical-walled relief with

        h(x, y) = lambda * phi(x, y) / (2 pi (n_doe - 1)),  phi in [0, 2 pi)

    giving a step of 0.366492 um and a total relief of 2.565443 um at the A1
    index.  This is the standard binary-optic realisation and is what makes the
    thin-element phase the design optimised for the phase the part delivers in
    the TEA limit.

A3. NOT 4 MASKS.  ``DAMMANN_PHASESTEPS = 4`` is NOT a mask/etch-step count: it
    is the annealing-schedule EXPONENT of ``makedammann2d`` (``phaselevelscur =
    phaselevels * 2**floor(phasesteps*(itr-it)/itr)``).  An 8-level binary optic
    is 3 masks.  No mask count is recorded anywhere, and none is needed here --
    the staircase is modelled as the 7 vertical-walled slices it is, which is
    EXACT for the piecewise-constant relief regardless of how it was patterned.

A4. WHICH WAY ROUND THE PLATE SITS -- NOT RECORDED, AND NOT DETERMINED BY THE
    ORDER TILTS.  The plate is a glass substrate with the relief on ONE face,
    and the beam can meet either face first:

        'relief_first'     air -> 7 slices -> glass substrate   (DEFAULT)
        'substrate_first'  glass substrate -> 7 slices -> air

    It is tempting to fix this from the pipeline's order tilts -- it assigns
    order ``(m, n)`` the tilt ``(m, n) lambda / period`` with NO index, so the
    orders travel in AIR -- but that does NOT discriminate: the plate's OTHER
    face is flat, and a flat face refracts the in-substrate tilts back to
    exactly ``m lambda / period`` in air.  Both mountings are consistent with
    the record and with the pipeline.  They are DIFFERENT rigorous problems,
    the choice is a keyed parameter (``build_stack(mount=...)``), and both are
    reported: measured, they differ by ~3.5 % in in-band efficiency at the
    highest truncation where both solve, and ``'substrate_first'`` loses the
    solver's energy guard one truncation earlier.  ``'relief_first'`` is the
    model of record here FOR THAT NUMERICAL REASON ALONE, which is stated
    rather than dressed up as physics.

    Either way the OTHER flat air/glass face is OUTSIDE the RCWA cell.  It is
    flat, so it applies a common Fresnel amplitude factor to every order and
    cannot redistribute them; the factor is reported by
    :func:`flat_face_fresnel` so a consumer can apply it, and it is deliberately
    NOT folded into the table (a 1 mm-thick plate modelled with both faces
    coherently is a 1 mm etalon, which is not the physics of a real wedged /
    incoherently-thick substrate).  Its per-order angular variation across this
    design's 0.046 rad order cone is ~1e-4 relative, so it is a scale and not a
    redistribution.

A5. THE CELL IS THE PIPELINE'S OWN.  The table is built on the SAME 128-pixel
    Dammann cell ``_d121_common.order_table`` uses (``cell_pixels=128``, seed
    42), not on the design tree's 174-pixel ``doe_cache`` variant, so the
    scalar-vs-RCWA comparison is on one geometry.  Pixel pitch 0.888724 um;
    minimum realised feature 1 pixel, i.e. SUB-WAVELENGTH at 1.31 um -- which
    is precisely why the thin-element assumption is worth testing.

A6. VERTICAL WALLS, NO ROUNDING, NO ETCH BIAS, NO ABSORPTION.  The reconstructed
    cell is the ideal staircase: real fabrication adds sidewall angle, corner
    rounding and a mask-to-mask overlay error, none of which is recorded and
    all of which would REDUCE the agreement with the scalar table further.  The
    RCWA-vs-scalar deltas here are therefore a LOWER BOUND on the physical
    deviation, not an estimate of it.

CONVENTIONS
===========
The library's public convention is ``exp(-i w t)``, so a wave travelling +z
carries ``exp(+i k z)`` and extra optical thickness ``(n-1) h`` advances the
phase by ``+k0 (n-1) h``.  A pillar of height ``h = lambda phi / (2 pi (n-1))``
therefore realises the thin-element transmittance ``exp(+i phi)`` -- which is
``nf`` itself, since ``makedammann2d`` returns ``nf = exp(1j * phase)``.  That
sign is not asserted: :func:`verify_relief_sign` MEASURES it on a uniform slab
(two cheap RCWA solves) and the table builder records the result.

THE REFERENCE PLANE, measured rather than assumed.  The solver's transmitted
order amplitude carries the FULL optical path through the stack (``k0 n d`` for
a uniform slab), NOT the excess over an equal air leg (``k0 (n-1) d``); the two
differ by exactly ``k0 d`` and the uniform-slab test in
``tests/unit/test_doe_rcwa.py`` pins the closed form.  Every order crosses the
same stack, so this is a GLOBAL PISTON on the table and it cancels out of every
inter-order phase -- which is why :func:`compare_to_scalar` reports the
piston-removed phase deltas as the headline and the raw piston separately.

THE PER-ORDER SCALAR
====================
A decomposer wants ONE complex number per order.  RCWA gives a 2-vector
``(tx, ty)`` per order per incident polarization.  The scalar taken here is the
CO-POLARIZED projection, renormalised so its squared modulus is the order's
true diffraction efficiency:

    a_m = sqrt(eta_m) * exp(i * arg(e_p . t_m))

so ``|a_m|**2`` is energy-exact (it carries the Poynting flux weight and the
longitudinal component that the raw tangential amplitude drops) while
``arg(a_m)`` is the co-polarized phase.  The cross-polarized FRACTION is
recorded alongside per order rather than discarded.
"""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

#: Bump to orphan every cached table by hand.
TABLE_SCHEMA = 1

#: Fused silica (Malitson) at 1.31 um -- assumption A1, not a recorded value.
N_FUSED_SILICA_1310 = 1.446804


# ===========================================================================
# the structure
# ===========================================================================
@dataclass(frozen=True)
class DoeStructure:
    """A multi-level surface-relief DOE cell, as RCWA sees it.

    Parameters
    ----------
    levels : (S, S) int ndarray
        Phase-level index ``0 .. n_levels-1`` per cell pixel.  Level ``k`` is
        the thin-element phase ``2 pi k / n_levels``.
    period : float
        Cell period along x and y (metres); square by construction here.
    wavelength : float
        Vacuum wavelength (metres).
    n_doe : float
        Refractive index of the relief material at ``wavelength``
        (ASSUMPTION A1).
    n_levels : int
        Number of phase levels (8 for design 121).
    relief_sign : int
        Phase-to-height sign; see :func:`relief_heights`.  ``+1`` (the
        ``exp(-i w t)`` convention, MEASURED by :func:`verify_relief_sign`) maps
        level ``k`` to a glass column of ``k`` steps; ``-1`` maps it to
        ``n_levels - 1 - k`` steps, realising the CONJUGATE design.  The two are
        different structures, so the sign is keyed.
    label : str
        Free-text provenance carried into the cache key.
    """

    levels: np.ndarray
    period: float
    wavelength: float
    n_doe: float = N_FUSED_SILICA_1310
    n_levels: int = 8
    relief_sign: int = 1
    label: str = ''

    def __post_init__(self):
        L = np.asarray(self.levels)
        if L.ndim != 2 or L.shape[0] != L.shape[1]:
            raise ValueError(
                f"DoeStructure.levels must be a SQUARE (S, S) level map; got "
                f"shape {L.shape}.")
        if not np.issubdtype(L.dtype, np.integer):
            raise ValueError(
                f"DoeStructure.levels must be an INTEGER level index map "
                f"(0 .. n_levels-1), not a phase in radians and not a complex "
                f"transmittance; got dtype {L.dtype}.  Use "
                f"levels_from_transmittance().")
        if L.min() < 0 or L.max() >= int(self.n_levels):
            raise ValueError(
                f"DoeStructure.levels range [{L.min()}, {L.max()}] is outside "
                f"[0, {int(self.n_levels) - 1}].")
        if int(self.relief_sign) not in (1, -1):
            raise ValueError("DoeStructure.relief_sign must be +1 or -1.")
        for name in ('period', 'wavelength', 'n_doe'):
            v = float(getattr(self, name))
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"DoeStructure.{name} must be finite and > 0; "
                                 f"got {v!r}.")
        if float(self.n_doe) <= 1.0:
            raise ValueError(
                f"DoeStructure.n_doe = {self.n_doe!r} <= 1: the relief depth "
                f"lambda*phi/(2 pi (n-1)) is not defined for a material that "
                f"does not exceed the surrounding air.")
        object.__setattr__(self, 'levels', np.ascontiguousarray(L, dtype=int))

    # -- derived geometry ---------------------------------------------------
    @property
    def n_cell(self) -> int:
        return int(self.levels.shape[0])

    @property
    def dz(self) -> float:
        """Relief height of ONE phase step (metres) -- assumption A2."""
        return float(self.wavelength) / (int(self.n_levels)
                                         * (float(self.n_doe) - 1.0))

    @property
    def relief_total(self) -> float:
        """Full relief height, level 0 to level ``n_levels-1`` (metres)."""
        return (int(self.n_levels) - 1) * self.dz

    @property
    def pixel(self) -> float:
        """Cell pixel pitch (metres)."""
        return float(self.period) / self.n_cell

    def eps_material(self) -> complex:
        return complex(float(self.n_doe) ** 2, 0.0)

    def thin_element_transmittance(self) -> np.ndarray:
        """The TEA transmittance this relief realises -- ``exp(i phi)``."""
        return np.exp(2j * np.pi * self.levels / int(self.n_levels))

    def key(self) -> Dict[str, Any]:
        """The structure's contribution to a cache key: geometry + a CONTENT
        hash of the level map (a 128x128 map is small, but hashing it keeps the
        key a fixed-size dict and catches a regenerated cell)."""
        h = hashlib.sha256(np.ascontiguousarray(self.levels,
                                                dtype=np.int64).tobytes())
        return {'levels_sha256': h.hexdigest(),
                'n_cell': self.n_cell,
                'period': repr(float(self.period)),
                'wavelength': repr(float(self.wavelength)),
                'n_doe': repr(float(self.n_doe)),
                'n_levels': int(self.n_levels),
                'relief_sign': int(self.relief_sign),
                'label': str(self.label)}


def levels_from_transmittance(nf, n_levels=8) -> np.ndarray:
    """Phase-level index map from a unit-modulus complex transmittance.

    ``makedammann2d`` returns ``nf = exp(1j * phase)`` with ``phase`` already
    quantised to multiples of ``2 pi / n_levels``; this recovers the integer
    index.  REFUSES a map that is not actually quantised, because a silently
    rounded level map is a silently different structure."""
    a = np.asarray(nf)
    if not np.iscomplexobj(a):
        raise ValueError(
            "levels_from_transmittance expects the COMPLEX transmittance "
            "makedammann2d returns (|nf| == 1), not a phase in radians.")
    mod = np.abs(a)
    if float(np.max(np.abs(mod - 1.0))) > 1e-9:
        raise ValueError(
            f"levels_from_transmittance: |nf| departs from 1 by "
            f"{float(np.max(np.abs(mod - 1.0))):.3e} -- this is not a "
            f"phase-only mask.")
    q = np.angle(a) * (int(n_levels) / (2.0 * np.pi))
    lv = np.rint(q).astype(int) % int(n_levels)
    resid = float(np.max(np.abs(q - np.rint(q))))
    if resid > 1e-6:
        raise ValueError(
            f"levels_from_transmittance: the mask is not quantised to "
            f"{n_levels} levels (worst level residual {resid:.3e}).  Refusing "
            f"to round a continuous phase into a staircase silently.")
    return lv


def design121_structure(cell_pixels=128, n_doe=N_FUSED_SILICA_1310,
                        relief_sign=1, wavelength=1.31e-6,
                        period=None, cache_dir=None) -> DoeStructure:
    """The design-121 DOE, reconstructed (assumptions A1-A6).

    Reads the SAME cached Dammann cell ``_d121_common.order_table`` uses.  If
    that cache is absent the design's ``.zmx`` is NOT required: only
    ``period`` is, and it defaults to the recorded 113.7566259645458 um.
    """
    per = 0.00011375662596454582 if period is None else float(period)
    cd = _HERE if cache_dir is None else str(cache_dir)
    fn = os.path.join(cd, f'_dammann_121_4x8_{int(cell_pixels)}.npy')
    if os.path.exists(fn):
        nf = np.load(fn)
    else:                                    # regenerate (deterministic)
        import lumenairy as la
        nf, _f, _c = la.makedammann2d(
            periodx=per, periody=per, waveln=wavelength,
            diforders=np.ones((4, 8)), phaselevels=8, phasesteps=4, itr=3000,
            seed=42, plot=False, cell_pixels=int(cell_pixels))
        np.save(fn, nf)
    return DoeStructure(levels=levels_from_transmittance(nf, 8), period=per,
                        wavelength=float(wavelength), n_doe=float(n_doe),
                        n_levels=8, relief_sign=int(relief_sign),
                        label=f'design121_dammann_4x8_{int(cell_pixels)}px')


def flat_face_fresnel(n_doe, theta=0.0):
    """Power transmittance of the plate's flat FRONT face (air -> substrate),
    which assumption A4 keeps OUTSIDE the RCWA cell.  Returned per polarization
    ``(T_s, T_p)`` so a consumer can restore the absolute scale."""
    n1, n2 = 1.0, float(n_doe)
    ct1 = np.cos(float(theta))
    st2 = np.sin(float(theta)) * n1 / n2
    ct2 = np.sqrt(max(0.0, 1.0 - st2 ** 2))
    rs = (n1 * ct1 - n2 * ct2) / (n1 * ct1 + n2 * ct2)
    rp = (n2 * ct1 - n1 * ct2) / (n2 * ct1 + n1 * ct2)
    return float(1.0 - rs ** 2), float(1.0 - rp ** 2)


# ===========================================================================
# the RCWA instrument
# ===========================================================================
def _cell_upsample_for(n_cell, n_orders):
    """Integer upsample factor so the cell clears the solver's Fourier-aliasing
    bound ``S > 4 * n_orders``.  EXACT: the relief is piecewise constant on the
    native lattice, so nearest-neighbour replication adds no geometry."""
    need = 4 * int(n_orders) + 1
    return max(1, int(np.ceil(need / float(n_cell))))


def relief_heights(struct: DoeStructure) -> np.ndarray:
    """Glass-column height in STEPS, per cell pixel, in the SOLVER's axis order.

    TWO CONVERSIONS HAPPEN HERE AND BOTH ARE LOAD-BEARING.

    1. THE AXIS ORDER IS TRANSPOSED.  The Dammann cell is stored the way
       ``_d121_common.order_table`` reads it -- ``A = fftshift(fft2(nf))`` is
       indexed ``A[my + cy, mx + cx]``, so ``nf`` has axis 0 = **y**, axis 1 =
       **x**.  ``rcwa_efficiency_2d``'s cell has the OPPOSITE convention:
       ``_eps_convolution_2d`` documents ``eps_cell[j, i]`` as the node
       ``(j Px/Sx, i Py/Sy)``, i.e. axis 0 = **x**.  Handing the cell over
       untransposed silently solves the TRANSPOSED structure and then reports
       order ``(m, n)``'s amplitude for the physical order ``(n, m)`` -- which
       on this design's 8-wide-by-4-tall order block moves half the requested
       orders out of band and reads ``sum |a|^2 = 0.4488`` against the true
       ``0.8851``.  It is energy-clean, it converges, and it is wrong: exactly
       the silent-wrong class this discipline exists to catch.
    2. THE PHASE-TO-HEIGHT SIGN.  ``relief_sign = +1`` maps phase level ``k`` to
       a glass column of ``k`` steps (extra optical thickness ADVANCES the
       phase, the ``exp(-i w t)`` convention that :func:`verify_relief_sign`
       measures); ``-1`` maps it to ``n_levels - 1 - k`` steps, which realises
       the CONJUGATE design.  Note this is NOT "pillars versus pits": a pillar
       array and the pit array that leaves the same column heights are the SAME
       solid, and RCWA sees one structure.
    """
    k = np.asarray(struct.levels).T                     # -> axis 0 = x
    return (k if int(struct.relief_sign) > 0
            else (int(struct.n_levels) - 1 - k))


#: Which face of the plate the beam meets first.  See :func:`build_stack`.
MOUNTS = ('relief_first', 'substrate_first')


def build_stack(struct: DoeStructure, n_orders, *, formulation='laurent',
                truncation='rectangular', cell_upsample=None,
                mount='relief_first'):
    """The ``n_levels - 1`` slice RCWA stack for the reconstructed relief.

    EXACT for the staircase: the relief is piecewise constant in ``z`` with
    vertical walls, so slice ``j`` (thickness ``dz``) is simply the set of
    pixels whose column reaches height ``j``.  No staircase approximation is
    involved and no slicing convergence parameter exists.

    ``mount`` -- WHICH WAY ROUND THE PLATE SITS.  NOT RECORDED anywhere in the
    design tree (assumption A4), and the two are DIFFERENT rigorous problems:

    * ``'relief_first'`` (default): air -> relief -> glass substrate.  The
      beam meets the etched face first; the diffracted orders travel in the
      substrate and are refracted back to ``m lambda / period`` in air by the
      flat BACK face.
    * ``'substrate_first'``: glass substrate -> relief -> air.  The beam
      enters the flat face; the orders leave the relief directly into air.

    The pipeline's order tilts (``m lambda / period``, no index) do NOT
    discriminate between them -- a flat exit face restores the air tilts either
    way -- so the record is genuinely silent.  Both are supported and the
    choice is KEYED.  What separates them here is numerical:
    ``'substrate_first'`` loses the solver's energy guard one truncation
    earlier (measured: unstable from ``n_orders = 6`` against ``8``), and the
    two disagree by ~3.5 % on the in-band efficiency at the highest truncation
    where both solve.  Both facts are reported rather than hidden.

    Either way ONE flat air/glass face is OUTSIDE the cell (the other side of
    the plate).  It is flat, so it cannot redistribute orders; see
    :func:`flat_face_fresnel`.
    """
    from lumenairy.elements.rcwa import RCWAStack
    if str(mount) not in MOUNTS:
        raise ValueError(f"build_stack: mount must be one of {MOUNTS}, got "
                         f"{mount!r}")
    up = (_cell_upsample_for(struct.n_cell, n_orders) if cell_upsample is None
          else int(cell_upsample))
    H = np.kron(relief_heights(struct), np.ones((up, up), dtype=int))
    nlv = int(struct.n_levels)
    eps_hi, eps_lo = struct.eps_material(), complex(1.0, 0.0)
    if str(mount) == 'relief_first':
        n_sup, n_sub, js = 1.0, float(struct.n_doe), range(nlv - 1, 0, -1)
    else:
        n_sup, n_sub, js = float(struct.n_doe), 1.0, range(1, nlv)
    st = RCWAStack(float(struct.period), period_y=float(struct.period),
                   n_superstrate=n_sup, n_substrate=n_sub,
                   n_orders=int(n_orders), n_orders_y=int(n_orders),
                   truncation=str(truncation))
    for j in js:
        st.add_layer(struct.dz,
                     eps_cell=np.where(H >= j, eps_hi, eps_lo).astype(complex),
                     formulation=str(formulation))
    return st


def _slab_zero_order_phase(n_doe, wavelength, thickness, n_orders=1,
                           period=None):
    """Transmitted zeroth-order phase of a uniform slab in air (one solve)."""
    from lumenairy.elements.rcwa import RCWAStack
    per = float(wavelength) * 8.0 if period is None else float(period)
    S = 4 * int(n_orders) + 1
    st = RCWAStack(per, period_y=per, n_superstrate=1.0, n_substrate=1.0,
                   n_orders=int(n_orders), n_orders_y=int(n_orders))
    st.add_layer(float(thickness),
                 eps_cell=np.full((S, S), complex(float(n_doe) ** 2)))
    res = st.set_source(float(wavelength), theta=0.0, phi=0.0).solve()
    m = res.per_order_amplitudes('transmission')
    o = np.asarray(m['orders'])
    i0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    return float(np.angle(np.asarray(m['Ex'])[0, i0]))


def verify_relief_sign(n_doe, wavelength, thickness=None, n_orders=1,
                       period=None, d_phase=0.5):
    """MEASURE the sign the whole relief construction rests on: does adding
    material ADVANCE the transmitted phase?

    Returns ``(d_measured, d_expected, sign)`` -- the measured and predicted
    phase CHANGE for a small increase in slab thickness, and ``+1`` when they
    agree in sign.

    WHY A DERIVATIVE AND NOT AN ABSOLUTE PHASE.  Two reasons, both learned
    from getting it wrong:

    * THE REFERENCE PLANE.  The solver's transmitted amplitude carries the FULL
      optical path through the stack, ``k0 n d``, not the EXCESS over an air
      leg, ``k0 (n-1) d`` (measured: the two differ by exactly ``k0 d``).  For
      the order table this is a global piston -- every order crosses the same
      stack -- but it makes an absolute-phase test test the wrong thing.
    * WRAPPING.  At the design-121 relief ``k0 n d = 17.8 rad``, so the ideal
      value and its negation are only separated modulo ``2 pi`` and can land
      arbitrarily close.  A derivative over a step chosen to move the phase by
      ``d_phase`` (default 0.5 rad, well inside a wrap) has no such ambiguity.

    The slab's own interface / Fabry-Perot phase is common to both thicknesses
    and largely cancels in the difference, which is the third reason.
    """
    k0 = 2.0 * np.pi / float(wavelength)
    t0 = (float(thickness) if thickness is not None
          else float(d_phase) / (k0 * float(n_doe)))
    dt = float(d_phase) / (k0 * float(n_doe))
    p0 = _slab_zero_order_phase(n_doe, wavelength, t0, n_orders, period)
    p1 = _slab_zero_order_phase(n_doe, wavelength, t0 + dt, n_orders, period)
    d_meas = float((p1 - p0 + np.pi) % (2 * np.pi) - np.pi)
    d_exp = float(k0 * float(n_doe) * dt)
    return d_meas, d_exp, (1 if d_meas * d_exp > 0 else -1)


def solve_orders(struct: DoeStructure, n_orders, *, theta=0.0, phi=0.0,
                 formulation='laurent', truncation='rectangular',
                 cell_upsample=None, want=None, stack=None,
                 on_unstable='stabilize', mount='relief_first'):
    """ONE RCWA solve -> per-order efficiencies + co-polarized complex scalars.

    Returns a dict with, for each requested order and each incident linear
    polarization (row 0 = incident ``E_x``, row 1 = incident ``E_y``):

    ``eta_T`` / ``eta_R``  the transmitted / reflected diffraction efficiency;
    ``amp``               the co-polarized complex scalar (``|amp|**2 ==
                          eta_T``, ``arg(amp)`` the co-polarized phase);
    ``xpol``              the cross-polarized POWER FRACTION of that order.

    Plus the energy accounting for the whole solve: ``sum_R``, ``sum_T`` and
    ``closure = sum_R + sum_T - 1`` per polarization.  A lossless dielectric
    cell must close; the value is REPORTED, never assumed.

    ``on_unstable`` -- what to do when the solver's OWN energy guard raises
    (``_EnergyError``).  This geometry sits squarely in the regime the guard's
    message names -- **very large period, low index contrast** (period is 86.8
    wavelengths and the contrast is 2.09:1) -- where an isolated ``n_orders``
    can hit a near-degenerate layer<->region mode match and blow up (measured:
    ``R+T = 2.1e+22``).  ``'stabilize'`` (default) RETRIES that solve with the
    library's own ``stabilize=True`` consensus search and RECORDS
    ``stabilized=True`` in the result; ``'raise'`` propagates.  A stabilized
    point is a point whose truncation is NOT the requested one, which is why it
    is flagged rather than absorbed.
    """
    st = (build_stack(struct, n_orders, formulation=formulation,
                      truncation=truncation, cell_upsample=cell_upsample,
                      mount=mount)
          if stack is None else stack)
    from lumenairy.elements.rcwa._core import _EnergyError
    st.set_source(float(struct.wavelength), theta=float(theta),
                  phi=float(phi))
    stabilized = False
    try:
        res = st.solve()
    except _EnergyError:
        if str(on_unstable) != 'stabilize':
            raise
        res = st.solve(stabilize=True)
        stabilized = True
    o_all, R, T = res.efficiencies()
    o_all = np.asarray(o_all)
    R = np.asarray(R)
    T = np.asarray(T)
    mod = res.per_order_amplitudes('transmission')
    tx = np.asarray(mod['Ex'])
    ty = np.asarray(mod['Ey'])

    if want is None:
        idx = np.arange(o_all.shape[0])
    else:
        w = [(int(a), int(b)) for a, b in want]
        pos = {(int(o_all[i, 0]), int(o_all[i, 1])): i
               for i in range(o_all.shape[0])}
        miss = [t for t in w if t not in pos]
        if miss:
            raise ValueError(
                f"solve_orders: order(s) {miss} are outside the retained set "
                f"at n_orders={n_orders} (truncation={truncation!r}).  Raise "
                f"n_orders; a design order that is not retained cannot be "
                f"reported as zero.")
        idx = np.array([pos[t] for t in w], dtype=int)

    # co-polarized projection: row 0 is the response to incident E_x (co-pol
    # component = tx), row 1 to incident E_y (co-pol = ty).
    co = np.stack([tx[0, idx], ty[1, idx]])          # (2, K)
    cross = np.stack([ty[0, idx], tx[1, idx]])       # (2, K)
    p_tan = np.abs(co) ** 2 + np.abs(cross) ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        xpol = np.where(p_tan > 0, np.abs(cross) ** 2 / p_tan, 0.0)
    etaT = T[:, idx]
    # |amp|^2 == the TRUE efficiency; the phase is the co-polarized one.  A
    # vanishing co-pol amplitude has no defined phase -- take 0 there rather
    # than a numerical-noise angle.
    ph = np.where(np.abs(co) > 0, np.angle(co), 0.0)
    amp = np.sqrt(np.maximum(etaT, 0.0)) * np.exp(1j * ph)

    sR = R.sum(axis=1)
    sT = T.sum(axis=1)
    return {'orders': o_all[idx],
            'eta_T': etaT, 'eta_R': R[:, idx], 'amp': amp, 'xpol': xpol,
            'sum_R': sR, 'sum_T': sT, 'closure': sR + sT - 1.0,
            'stabilized': bool(stabilized),
            'n_retained': int(o_all.shape[0]),
            'n_orders': int(n_orders), 'theta': float(theta),
            'phi': float(phi), 'formulation': str(formulation),
            'truncation': str(truncation), 'mount': str(mount)}


# ===========================================================================
# the scalar (thin-element) table, for comparison
# ===========================================================================
def scalar_table(struct: DoeStructure, want=None):
    """The THIN-ELEMENT table this module replaces -- ``fft2`` of the cell's
    complex transmittance, in ``_d121_common.order_table``'s own convention
    (``A = fftshift(fft2(nf)) / nf.size``, so ``A[m]`` multiplies
    ``exp(+i m G x)``).  Reproduced here rather than imported so the comparison
    does not need the ``.zmx``; :func:`assert_matches_d121_order_table` pins
    the two together."""
    nf = struct.thin_element_transmittance()
    A = np.fft.fftshift(np.fft.fft2(nf)) / nf.size
    S = struct.n_cell
    c = S // 2
    if want is None:
        return None, A
    o = np.array([(int(a), int(b)) for a, b in want], dtype=int)
    return o, A[o[:, 1] + c, o[:, 0] + c]


def design_order_set(struct: DoeStructure, n_x=8, n_y=4):
    """The design's own 32 order indices, selected the way
    ``_d121_common.order_table`` selects them (the ``n_x * n_y`` strongest DFT
    peaks, lexsorted by ``(my, mx)``)."""
    _o, A = scalar_table(struct)
    S = struct.n_cell
    c = S // 2
    P = np.abs(A) ** 2
    flat = np.argsort(P.ravel())[::-1][:int(n_x) * int(n_y)]
    oy, ox = np.unravel_index(flat, P.shape)
    mx, my = ox - c, oy - c
    k = np.lexsort((mx, my))
    return np.stack([mx[k], my[k]], axis=1)


# ===========================================================================
# angle grid
# ===========================================================================
def chebyshev_theta(theta_max, n_theta):
    """Chebyshev-Gauss-Lobatto nodes on ``[0, theta_max]``, ascending.

    WHY CHEBYSHEV, NOT UNIFORM.  The per-order amplitude ``a_m(theta)`` is
    analytic across this cone (the nearest Rayleigh anomaly is decades away --
    see the module doc's angular-domain note), and the quantity the decomposer
    needs is a WEIGHTED INTEGRAL of it.  Chebyshev nodes carry
    Clenshaw-Curtis quadrature, which converges spectrally for an analytic
    integrand where the trapezoid rule on a uniform grid converges as
    ``O(h**2)``; the same nodes also give a well-conditioned barycentric
    interpolant, which is what a future ANGLE-RESOLVED decomposition needs.
    The endpoint clustering is a bonus: the cone edge is where the amplitudes
    move most, and it is where a uniform grid is thinnest in information.
    """
    n = int(n_theta)
    if n < 1:
        raise ValueError("chebyshev_theta: n_theta must be >= 1.")
    if n == 1:
        return np.array([0.0])
    k = np.arange(n)
    u = np.cos(np.pi * k / (n - 1))              # +1 .. -1
    return np.sort(float(theta_max) * 0.5 * (1.0 - u))


def clenshaw_curtis_weights(n):
    """Clenshaw-Curtis weights for :func:`chebyshev_theta`'s nodes on [0, 1],
    ASCENDING (matching the node order this module returns)."""
    n = int(n)
    if n == 1:
        return np.array([1.0])
    N = n - 1
    w = np.zeros(n)
    for k in range(n):
        s = 0.0
        for j in range(1, N // 2 + 1):
            b = 1.0 if j == N / 2.0 else 2.0
            s += b / (4.0 * j * j - 1.0) * np.cos(2.0 * j * np.pi * k / N)
        c = 1.0 if (k == 0 or k == N) else 2.0
        w[k] = c / N * (1.0 - s)
    return w * 0.5            # map from [-1, 1] to [0, 1]


def angle_grid(theta_max, n_theta, n_phi):
    """``(thetas, phis, weight)`` for a disk in incident direction cosines.

    ``theta`` on Chebyshev nodes of ``[0, theta_max]``; ``phi`` uniform on
    ``[0, 2 pi)`` (the trapezoid rule is SPECTRALLY accurate for a periodic
    integrand, so uniform is the right choice there and Chebyshev is not).
    ``weight`` is the product quadrature weight for
    ``INT f(theta, phi) g(theta) theta dtheta dphi`` with the beam weight
    ``g`` supplied separately -- the ``theta`` Jacobian is folded in here.
    """
    th = chebyshev_theta(theta_max, n_theta)
    ph = (np.arange(int(n_phi)) * (2.0 * np.pi / int(n_phi))
          if int(n_phi) > 1 else np.array([0.0]))
    if len(th) == 1:
        # DEGENERATE GRID: one polar node is normal incidence only, and the
        # disk quadrature's own ``theta`` Jacobian would weight it ZERO -- a
        # table that then averaged to 0/0.  A single node is a POINT
        # EVALUATION, so it takes unit weight.  (This is the n_theta = 1
        # configuration a caller uses to ask for the normal-incidence table.)
        return th, ph, np.full((1, len(ph)), 1.0 / len(ph))
    wq = clenshaw_curtis_weights(len(th)) * float(theta_max)
    W = np.outer(wq * th, np.full(len(ph), 1.0 / len(ph)))
    return th, ph, W


# ===========================================================================
# the table + its cache
# ===========================================================================
_TABLE_FIELDS = ('orders', 'thetas', 'phis', 'amp', 'eta_T', 'eta_R', 'xpol',
                 'sum_R', 'sum_T', 'closure', 'quad_weight', 'stabilized')


def _lumenairy_source_sha():
    """SHA-256 over every ``.py`` of the lumenairy package actually imported.

    The SAME discipline as ``_d121_common._lumenairy_source_sha`` and for the
    same reason: a library default flip WITHIN one version changes this table
    and can never appear in a hand-spelled filename."""
    import lumenairy as la
    root = os.path.dirname(os.path.abspath(la.__file__))
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != '__pycache__')
        for name in sorted(filenames):
            if not name.endswith('.py'):
                continue
            p = os.path.join(dirpath, name)
            h.update(os.path.relpath(p, root).replace('\\', '/').encode())
            with open(p, 'rb') as fh:
                h.update(fh.read())
    return h.hexdigest()


def _self_sha():
    try:
        with open(os.path.abspath(__file__), 'rb') as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return '<unreadable:doe_rcwa_table.py>'


def table_key(struct: DoeStructure, orders, n_orders, thetas, phis, *,
              formulation, truncation, cell_upsample, mount='relief_first'):
    """The full parameter dict this table is keyed on, and its digest.

    The SAME discipline as ``_d121_common._chain_a_key`` (defect D6): the
    structure, the wavelength, the truncation AND the angle grid, plus a
    content hash of the library and of this module, so any edit that can change
    a number orphans the cache instead of serving a stale one."""
    import lumenairy as la
    key = {
        'schema': int(TABLE_SCHEMA),
        'lumenairy_version': str(la.__version__),
        'lumenairy_source_sha256': _lumenairy_source_sha(),
        'builder_sha256': _self_sha(),
        'structure': struct.key(),
        'orders': [[int(a), int(b)] for a, b in np.asarray(orders)],
        'n_orders': int(n_orders),
        'formulation': str(formulation),
        'truncation': str(truncation),
        'mount': str(mount),
        'cell_upsample': (None if cell_upsample is None else int(cell_upsample)),
        'thetas': [repr(float(t)) for t in np.asarray(thetas).ravel()],
        'phis': [repr(float(p)) for p in np.asarray(phis).ravel()],
    }
    blob = json.dumps(key, sort_keys=True, separators=(',', ':'))
    return key, hashlib.sha256(blob.encode('ascii')).hexdigest()


def table_path(digest, cache_dir=None):
    cd = _HERE if cache_dir is None else str(cache_dir)
    return os.path.join(cd, f'_doe_rcwa_v{TABLE_SCHEMA}_{digest[:16]}.npz')


def build_table(struct: DoeStructure, orders, n_orders, thetas, phis, *,
                formulation='laurent', truncation='rectangular',
                cell_upsample=None, quad_weight=None, cache=True,
                cache_dir=None, max_workers=None, blas_per_worker=1,
                mount='relief_first', log=print):
    """Solve the angle x polarization sweep and cache the order table.

    Returns a dict with, per angle point ``(i_theta, i_phi)`` and per incident
    polarization, the complex co-polarized amplitude of every requested order,
    plus the per-solve energy accounting.  Shapes:

        ``amp``    (n_theta, n_phi, 2, K) complex
        ``eta_T``  (n_theta, n_phi, 2, K) float      ``eta_R`` same
        ``xpol``   (n_theta, n_phi, 2, K) float
        ``sum_R`` / ``sum_T`` / ``closure``  (n_theta, n_phi, 2) float

    THREADING follows ``RCWAStack.solve_vs_wavelength``'s own pattern and for
    its own reason (M4 2026-08-04): the solves are independent and NumPy
    releases the GIL inside LAPACK, but the BLAS cap is PROCESS-GLOBAL on
    OpenBLAS, so it is applied ONCE around the whole sweep on the calling
    thread rather than per worker.  Results are stored BY INDEX, so completion
    order is irrelevant.
    """
    from concurrent.futures import ThreadPoolExecutor

    from lumenairy.elements.rcwa._core import _blas_limit, _blas_threads_quiet

    orders = np.asarray([[int(a), int(b)] for a, b in np.asarray(orders)],
                        dtype=int)
    th = np.atleast_1d(np.asarray(thetas, dtype=float))
    ph = np.atleast_1d(np.asarray(phis, dtype=float))
    key, digest = table_key(struct, orders, n_orders, th, ph,
                            formulation=formulation, truncation=truncation,
                            cell_upsample=cell_upsample, mount=mount)
    key_blob = json.dumps(key, sort_keys=True, separators=(',', ':'))
    fn = table_path(digest, cache_dir)
    if cache and os.path.exists(fn):
        d = np.load(fn, allow_pickle=False)
        stored = str(d['key_json']) if 'key_json' in d.files else None
        if stored != key_blob:
            raise RuntimeError(
                f"build_table: {os.path.basename(fn)} does not carry the cache "
                f"key its name claims.  This file was not written by this "
                f"configuration -- delete it rather than trust it.")
        out = {k: d[k] for k in _TABLE_FIELDS if k in d.files}
        out['key'] = key
        out['digest'] = digest
        out['path'] = fn
        out['meta'] = json.loads(str(d['meta']))
        return out

    K = orders.shape[0]
    nt, np_ = len(th), len(ph)
    amp = np.full((nt, np_, 2, K), np.nan, dtype=complex)
    etaT = np.full((nt, np_, 2, K), np.nan)
    etaR = np.full((nt, np_, 2, K), np.nan)
    xpol = np.full((nt, np_, 2, K), np.nan)
    sR = np.full((nt, np_, 2), np.nan)
    sT = np.full((nt, np_, 2), np.nan)
    stab = np.zeros((nt, np_), dtype=bool)
    up = (_cell_upsample_for(struct.n_cell, n_orders)
          if cell_upsample is None else int(cell_upsample))

    jobs = [(i, j) for i in range(nt) for j in range(np_)]
    # theta == 0 is azimuth-degenerate: every phi is the same physical
    # incidence.  Solve it ONCE and broadcast (this is not an approximation --
    # the incident wavevector is identical).
    zero = [k for k, (i, j) in enumerate(jobs)
            if abs(th[i]) < 1e-15 and j > 0]
    jobs = [jb for k, jb in enumerate(jobs) if k not in set(zero)]

    def _one(ij):
        i, j = ij
        st = build_stack(struct, n_orders, formulation=formulation,
                         truncation=truncation, cell_upsample=up, mount=mount)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            r = solve_orders(struct, n_orders, theta=float(th[i]),
                             phi=float(ph[j]), formulation=formulation,
                             truncation=truncation, cell_upsample=up,
                             want=orders, stack=st, mount=mount)
        r['warnings'] = [str(x.message)[:200] for x in w][:8]
        return i, j, r

    warn_log = []

    def _store(i, j, r):
        amp[i, j] = r['amp']
        etaT[i, j] = r['eta_T']
        etaR[i, j] = r['eta_R']
        xpol[i, j] = r['xpol']
        sR[i, j] = r['sum_R']
        sT[i, j] = r['sum_T']
        stab[i, j] = bool(r.get('stabilized', False))
        for m in r.get('warnings', ()):
            warn_log.append(f"theta={th[i]:.6e} phi={ph[j]:.6e}: {m}")

    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, max(1, len(jobs)))
    max_workers = max(1, int(max_workers))
    import time as _time
    t0 = _time.perf_counter()
    with _blas_threads_quiet(blas_per_worker), _blas_limit():
        if max_workers == 1 or len(jobs) == 1:
            for ij in jobs:
                _store(*_one(ij))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for i, j, r in ex.map(_one, jobs):
                    _store(i, j, r)
    wall = _time.perf_counter() - t0
    for i in range(nt):                       # broadcast the theta == 0 row
        if abs(th[i]) < 1e-15 and np_ > 1:
            for j in range(1, np_):
                amp[i, j] = amp[i, 0]
                etaT[i, j] = etaT[i, 0]
                etaR[i, j] = etaR[i, 0]
                xpol[i, j] = xpol[i, 0]
                sR[i, j] = sR[i, 0]
                sT[i, j] = sT[i, 0]
                stab[i, j] = stab[i, 0]

    qw = (np.ones((nt, np_)) if quad_weight is None
          else np.asarray(quad_weight, dtype=float).reshape(nt, np_))
    meta = {'n_solves': len(jobs), 'wall_s': float(wall), 'mount': str(mount),
            'max_workers': int(max_workers), 'cell_upsample': int(up),
            'n_stabilized': int(stab.sum()),
            'worst_closure': float(np.nanmax(np.abs(sR + sT - 1.0))),
            'relief_total_m': float(struct.relief_total),
            'dz_m': float(struct.dz), 'pixel_m': float(struct.pixel),
            'flat_face_fresnel': list(flat_face_fresnel(struct.n_doe)),
            'warnings': warn_log[:40]}
    out = {'orders': orders, 'thetas': th, 'phis': ph, 'amp': amp,
           'eta_T': etaT, 'eta_R': etaR, 'xpol': xpol, 'sum_R': sR,
           'sum_T': sT, 'closure': sR + sT - 1.0, 'quad_weight': qw,
           'stabilized': stab}
    if cache:
        # the temp name MUST already end in '.npz': savez_compressed appends
        # the suffix itself when it is absent, so 'x.npz.tmp' is written as
        # 'x.npz.tmp.npz' and the os.replace below then fails on a name that
        # was never created.
        tmp = fn + '.tmp.npz'
        np.savez_compressed(tmp, key_json=np.array(key_blob),
                            meta=np.array(json.dumps(meta)), **out)
        os.replace(tmp, fn)
    out['key'] = key
    out['digest'] = digest
    out['path'] = fn
    out['meta'] = meta
    if log:
        log(f"  [table] {len(jobs)} solves, {wall:.1f} s, worst |R+T-1| "
            f"{meta['worst_closure']:.3e} -> {os.path.basename(fn)}")
    return out


def beam_weighted_amplitudes(table, beam_weight=None):
    """The per-order scalar the decomposer uses: the BEAM-WEIGHTED,
    POLARIZATION-AVERAGED complex amplitude.

    THE AVERAGING CHOICE, stated because it is a modelling decision and not a
    detail.  The pipeline's beam basis carries ONE complex weight per order, so
    an angle-resolved table has to be collapsed.  Two collapses are defensible
    and they are NOT the same number:

      * COHERENT (taken here): ``a_bar = INT w(theta,phi) a(theta,phi) /
        INT w``.  This is the amplitude of the beam-averaged FIELD, and it is
        the one that is correct to first order for the pipeline's own
        construction -- the decomposer's beams are added COHERENTLY, so the
        quantity that must survive the collapse is the field, not the power.
        It under-reports power when the phase varies across the cone (the
        variation is then real decoherence, not a modelling loss).
      * INCOHERENT: ``sqrt(INT w |a|**2 / INT w)``, which preserves power and
        discards the phase spread.

    Both are returned (``amp_coherent`` / ``amp_incoherent``) together with
    ``coherence`` = ``|amp_coherent|**2 / amp_incoherent**2``, which MEASURES
    how much the choice matters -- it is 1 exactly when the collapse is
    lossless.  Full ANGLE-RESOLVED decomposition (one beam per order per angle
    node) is a stated future refinement; it needs a pipeline beam basis that
    carries an angle, which does not exist today.

    Polarization is averaged as an UNPOLARIZED source (equal-weight mean over
    the two incident linear states).  For a design whose source polarization is
    known, pass the per-polarization rows through instead -- they are in the
    table.
    """
    amp = np.asarray(table['amp'])                   # (nt, nph, 2, K)
    W = np.asarray(table['quad_weight'], dtype=float)
    if beam_weight is not None:
        W = W * np.asarray(beam_weight, dtype=float).reshape(W.shape)
    tot = float(W.sum())
    if not np.isfinite(tot) or tot <= 0:
        raise ValueError("beam_weighted_amplitudes: the angular weight sums to "
                         f"{tot!r}; nothing to average.")
    w4 = W[:, :, None, None]
    coh = (amp * w4).sum(axis=(0, 1)) / tot          # (2, K)
    inc = np.sqrt((np.abs(amp) ** 2 * w4).sum(axis=(0, 1)) / tot)
    coh_pol = coh.mean(axis=0)
    inc_pol = np.sqrt((inc ** 2).mean(axis=0))
    with np.errstate(invalid='ignore', divide='ignore'):
        q = np.where(inc_pol > 0, np.abs(coh_pol) ** 2 / inc_pol ** 2, 1.0)
    return {'amp_coherent': coh_pol, 'amp_incoherent': inc_pol,
            'coherence': q, 'amp_per_pol_coherent': coh,
            'amp_per_pol_incoherent': inc,
            'orders': np.asarray(table['orders'])}


def gaussian_beam_angular_weight(thetas, phis, theta_rms):
    """Angular power weight of a rotationally symmetric beam whose angular
    spectrum has RMS half-angle ``theta_rms``:
    ``w = exp(-theta**2 / theta_rms**2)``, uniform in ``phi``.

    Used as the DEFAULT beam weight because design 121's field at the DOE is
    measured rotationally symmetric to within its own tail; a caller with the
    measured 2-D angular spectrum can pass it directly."""
    th = np.atleast_1d(np.asarray(thetas, dtype=float))
    nph = len(np.atleast_1d(np.asarray(phis)))
    g = np.exp(-(th / float(theta_rms)) ** 2)
    return np.repeat(g[:, None], nph, axis=1)


# ===========================================================================
# convergence ladder
# ===========================================================================
def convergence_ladder(struct: DoeStructure, orders, n_orders_list, *,
                       theta=0.0, phi=0.0, formulation='laurent',
                       truncation='rectangular', mount='relief_first',
                       on_unstable='stabilize', log=print):
    """Solve the design orders at a ladder of truncations and report the
    RUNG-TO-RUNG movement of the order amplitudes.

    The convergence criterion is on the OBSERVABLE the table exports -- the
    complex per-order amplitude -- not on the total power, because a lossless
    cell conserves energy at EVERY truncation and total power therefore proves
    nothing about the per-order split (the lossless trap).

    Returns a list of dicts, one per rung, each carrying the amplitudes, the
    energy closure, the wall time, and (from the second rung on) the worst
    per-order deltas against the previous rung.
    """
    import time as _time
    rows = []
    prev = None
    for M in n_orders_list:
        t0 = _time.perf_counter()
        r = solve_orders(struct, int(M), theta=theta, phi=phi,
                         formulation=formulation, truncation=truncation,
                         mount=mount, on_unstable=on_unstable, want=orders)
        dt = _time.perf_counter() - t0
        row = {'n_orders': int(M), 'wall_s': float(dt),
               'n_retained': int(r['n_retained']),
               'amp': r['amp'], 'eta_T': r['eta_T'],
               'closure': r['closure'], 'sum_T': r['sum_T'],
               'xpol': r['xpol'], 'stabilized': bool(r['stabilized'])}
        if prev is not None:
            d_eta = np.abs(row['eta_T'] - prev['eta_T'])
            d_amp = np.abs(row['amp'] - prev['amp'])
            dph = np.angle(row['amp'] * np.conj(prev['amp']))
            # piston-removed phase movement: a uniform phase shift between
            # rungs is not a convergence failure of the RELATIVE order phases,
            # which is what a decomposer consumes.
            pis = np.angle((row['amp'] * np.conj(prev['amp'])).sum(axis=1))
            dph_rel = np.angle(np.exp(1j * (dph - pis[:, None])))
            row.update(d_eta_max=float(np.max(d_eta)),
                       d_amp_max=float(np.max(d_amp)),
                       d_phase_max=float(np.max(np.abs(dph))),
                       d_phase_rel_max=float(np.max(np.abs(dph_rel))),
                       d_sumT=float(np.max(np.abs(row['sum_T']
                                                  - prev['sum_T']))))
        rows.append(row)
        if log:
            tail = ('' if prev is None else
                    f"  d_eta {row['d_eta_max']:.3e}  d|a| {row['d_amp_max']:.3e}"
                    f"  d_arg(rel) {row['d_phase_rel_max']:.3e} rad")
            log(f"  M={M:3d}  N={row['n_retained']:5d}  {dt:8.1f} s  "
                f"sumT {row['sum_T'][0]:.6f}  |R+T-1| "
                f"{np.max(np.abs(row['closure'])):.2e}{tail}")
        prev = row
    return rows


# ===========================================================================
# scalar-vs-RCWA comparison
# ===========================================================================
def compare_to_scalar(struct: DoeStructure, orders, amp_rcwa, total_T=None):
    """Per-order efficiency and phase deltas, RCWA against the thin-element
    table.

    The PHASE comparison is reported twice.  The raw difference carries a
    global piston that is physically meaningless (RCWA references the incident
    field at the top of the stack; the TEA table references the cell), so the
    headline number is the PISTON-REMOVED per-order phase delta -- the quantity
    a coherent decomposer actually consumes.  The raw piston is reported too,
    because a reader who wants to know it should not have to re-derive it.

    ``total_T`` -- the solve's total transmitted power over ALL retained
    orders.  When given, the comparison ALSO reports the SPLITTING RATIO
    ``frac = eta / total_T`` against the scalar table's own (the scalar
    transmittance is unit-modulus, so its total is 1).  This separates the two
    things a rigorous solve changes and the thin-element model conflates:
    THROUGHPUT (Fresnel reflection at the relief, which the ideal phase screen
    does not have at all) and REDISTRIBUTION (power moved between orders).
    Only the second is a statement about the DESIGN.
    """
    o, a_s = scalar_table(struct, want=orders)
    a_r = np.asarray(amp_rcwa).ravel()
    if a_r.shape != a_s.shape:
        raise ValueError(f"compare_to_scalar: RCWA amplitudes have shape "
                         f"{a_r.shape}, scalar table {a_s.shape}.")
    eta_s = np.abs(a_s) ** 2
    eta_r = np.abs(a_r) ** 2
    piston = float(np.angle(np.sum(a_r * np.conj(a_s))))
    dphi = np.angle(a_r * np.conj(a_s))
    dphi_rel = np.angle(np.exp(1j * (dphi - piston)))
    with np.errstate(invalid='ignore', divide='ignore'):
        rel = np.where(eta_s > 0, (eta_r - eta_s) / eta_s, np.nan)
    out = {'orders': np.asarray(orders), 'eta_scalar': eta_s,
           'eta_rcwa': eta_r, 'd_eta': eta_r - eta_s, 'rel_eta': rel,
           'piston': piston, 'd_phase': dphi, 'd_phase_rel': dphi_rel,
           'sum_scalar': float(eta_s.sum()), 'sum_rcwa': float(eta_r.sum()),
           'uniformity_scalar': float(eta_s.min() / eta_s.max()),
           'uniformity_rcwa': float(eta_r.min() / eta_r.max())}
    if total_T is not None:
        tt = float(total_T)
        if not np.isfinite(tt) or tt <= 0:
            raise ValueError(f"compare_to_scalar: total_T must be > 0, got "
                             f"{total_T!r}.")
        out['frac_rcwa'] = eta_r / tt
        out['frac_scalar'] = eta_s
        out['d_frac'] = eta_r / tt - eta_s
        out['throughput'] = tt
        out['sum_frac_rcwa'] = float(eta_r.sum() / tt)
    return out


def assert_matches_d121_order_table(struct: DoeStructure, atol=1e-12):
    """Prove :func:`scalar_table` + :func:`design_order_set` reproduce
    ``_d121_common.order_table`` EXACTLY (needs the design-121 module; used by
    the CLI and by the doc, not by the tests, which must run without it)."""
    import sys
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import _d121_common as C
    mx, my, amps = C.order_table(float(struct.period), n_per=struct.n_cell)
    o = design_order_set(struct)
    _o, a = scalar_table(struct, want=o)
    ok_o = bool(np.array_equal(o[:, 0], mx) and np.array_equal(o[:, 1], my))
    d = float(np.max(np.abs(a - amps)))
    if not ok_o or d > atol:
        raise AssertionError(
            f"scalar_table does not reproduce _d121_common.order_table "
            f"(orders match: {ok_o}, worst |da| = {d:.3e}).")
    return d


# ===========================================================================
# CLI -- the three things a study does with this module
# ===========================================================================
def _cli(argv=None):
    """``ladder`` / ``sweep`` / ``compare`` on the design-121 cell.

    Named in :func:`build_table`'s own refusal message, so a run that declines
    to spend an RCWA sweep inside its decompose stage has a command to point
    at.  BLAS is pinned to one thread per worker for the same reason
    ``RCWAStack.solve_vs_wavelength`` does it: these are many independent
    medium solves, and an oversubscribed OpenBLAS pool makes each of them
    hundreds of times slower (measured on this box: a 162-square complex
    inverse went 7.5 s -> 0.021 s under the cap).
    """
    import argparse
    ap = argparse.ArgumentParser(prog='doe_rcwa_table', description=__doc__)
    ap.add_argument('action', choices=('ladder', 'sweep', 'compare'))
    ap.add_argument('--n-doe', type=float, default=N_FUSED_SILICA_1310)
    ap.add_argument('--relief-sign', type=int, default=1, choices=(1, -1))
    ap.add_argument('--mount', default='relief_first', choices=MOUNTS)
    ap.add_argument('--cell-pixels', type=int, default=128)
    ap.add_argument('--n-orders', type=int, default=6)
    ap.add_argument('--ladder', default='4,5,6',
                    help='comma-separated n_orders for the ladder')
    ap.add_argument('--theta-max', type=float, default=5.0e-4)
    ap.add_argument('--n-theta', type=int, default=1)
    ap.add_argument('--n-phi', type=int, default=1)
    ap.add_argument('--formulation', default='laurent')
    ap.add_argument('--truncation', default='rectangular')
    ap.add_argument('--workers', type=int, default=None)
    a = ap.parse_args(argv)

    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(v, '1')
    struct = design121_structure(cell_pixels=a.cell_pixels, n_doe=a.n_doe,
                                 relief_sign=a.relief_sign)
    orders = design_order_set(struct)
    print(f"design-121 cell: period {struct.period * 1e6:.6f} um, "
          f"{struct.n_cell} px ({struct.pixel * 1e6:.6f} um = "
          f"{struct.pixel / struct.wavelength:.3f} lambda), n_doe "
          f"{struct.n_doe:.6f}, dz {struct.dz * 1e6:.6f} um, relief "
          f"{struct.relief_total * 1e6:.6f} um, mount {a.mount}")
    d_meas, d_exp, sign = verify_relief_sign(struct.n_doe, struct.wavelength)
    print(f"relief-sign measurement: d_phase {d_meas:+.6f} rad vs predicted "
          f"{d_exp:+.6f} -> sign {sign:+d}")

    if a.action == 'ladder':
        Ms = [int(x) for x in a.ladder.split(',')]
        convergence_ladder(struct, orders, Ms, mount=a.mount,
                           formulation=a.formulation, truncation=a.truncation,
                           on_unstable='raise')
        return 0

    th, ph, quad = angle_grid(a.theta_max, a.n_theta, a.n_phi)
    tab = build_table(struct, orders, a.n_orders, th, ph, quad_weight=quad,
                      formulation=a.formulation, truncation=a.truncation,
                      mount=a.mount, max_workers=a.workers)
    print(f"  cached at {tab['path']}")
    print(f"  {tab['meta']['n_solves']} solves, "
          f"{tab['meta']['n_stabilized']} stabilized, worst |R+T-1| "
          f"{tab['meta']['worst_closure']:.3e}")
    if a.action == 'compare':
        avg = beam_weighted_amplitudes(tab)
        c = compare_to_scalar(struct, orders, avg['amp_coherent'],
                              total_T=float(np.nanmean(tab['sum_T'])))
        print(f"\n  in-band sum|a|^2   RCWA {c['sum_rcwa']:.6f}   scalar "
              f"{c['sum_scalar']:.6f}   ({100 * (c['sum_rcwa'] / c['sum_scalar'] - 1):+.1f} %)")
        print(f"  in-band / sum T    {c['sum_frac_rcwa']:.6f}   (throughput "
              f"{c['throughput']:.6f})")
        print(f"  uniformity         RCWA {c['uniformity_rcwa']:.6f}   scalar "
              f"{c['uniformity_scalar']:.6f}")
        print(f"  phase: piston {c['piston']:+.4f} rad; worst piston-removed "
              f"{np.max(np.abs(c['d_phase_rel'])):.4f} rad; rms "
              f"{np.sqrt(np.mean(c['d_phase_rel'] ** 2)):.4f} rad")
        print(f"  max cross-pol fraction {float(np.max(tab['xpol'])):.3e}")
        print(f"\n  {'order':>8}  {'eta_scalar':>11}  {'eta_rcwa':>11}  "
              f"{'ratio':>8}  {'d_arg(rel)':>11}")
        for i, (m, n) in enumerate(np.asarray(orders)):
            print(f"  ({m:+d},{n:+d})  {c['eta_scalar'][i]:11.6f}  "
                  f"{c['eta_rcwa'][i]:11.6f}  "
                  f"{c['eta_rcwa'][i] / c['eta_scalar'][i]:8.4f}  "
                  f"{c['d_phase_rel'][i]:+11.4f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(_cli())

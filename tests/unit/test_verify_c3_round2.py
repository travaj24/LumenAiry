"""VERIFY-WP-C3 ROUND 2 -- the one decision this re-verification had to take.

WP-C3 round 2 left ONE question open and pinned a test id around it
(``tests/unit/test_niche_p8_capstone.py::test_stepB_composed_doublet_relay_
matches_debye``): on the composed doublet + relay chain the Collins final leg
reads EE80 = 12.3588 um against the co-moving step's 11.0139 um, the test's
ring-Huygens Debye oracle backs the co-moving column, and round 2 could not
say whether the Collins FIELD is wrong or only the LATTICE it returns is too
coarse to measure EE80 on.

IT IS THE LATTICE, and this module is where that is asserted rather than
described.  MEASURED 2026-09-20 on the capstone's own chain, WIN-py3.14 and
WSL-py3.12 agreeing to every printed digit
(``validation/probe_verify_c3_round2/vr2_p8_settle_{win,wsl}.json``):

* the two arms' chain PREFIX is bit-identical (``max|env3_collins -
  env3_sziklas| = 0.0``), so the only difference is the final leg;
* read on their own returned lattices, EE80 is 11.0139 um (co-moving, pitch
  0.1052 um, 105 samples inside EE80) against 12.3588 um (Collins, pitch
  floored at 6.3126 um, **1.96** samples inside EE80) -- 11.24 % and -0.87 %
  from the oracle's 11.1102 um;
* resampled onto ONE common 0.5 um lattice by exact full-N band-limited
  interpolation, the same two fields read EE80 **11.0575 um** and **11.0568
  um** -- they agree to **0.006 %** -- and their intensity maps agree to
  **1.29e-04** in relative L2;
* the same Collins leg driven with an explicit ``dx_out`` reads EE80
  11.0090 / 10.9921 / 11.0708 / 10.9417 / 10.9083 um at 0.25 / 0.5 / 1 / 2 /
  3 um and only 12.3588 um at its own floored 6.3126 um.

So the Collins field on that leg is right to 1.3e-04 and the 11.2 % is the
metric quantising on a two-sample lattice.  The library says so itself: the
shipped leg emits ``the Collins chirp-Z stage is under-sampled -- K2 (output)
5.1055``.

WHY THIS MODULE DOES NOT RUN THAT CHAIN.  The capstone composition is a
4096-grid traced doublet plus a ring-Huygens oracle; it is minutes, and it is
already run (as a ``slow`` id) by the capstone itself.  The id below
reproduces the SHAPE of that leg -- a flat-resolving Collins leg whose chirp-Z
is representable, whose output pitch is therefore set by the floor ``2 r_out /
N`` rather than by the co-moving contraction -- on a 256 grid in under a
second, and states the separation as a comparison between readings of the SAME
transport, which is the form that carries no cross-build spread.

Bars and their measured values, both builds
(``validation/probe_verify_c3_round2/vr2_p8_decision_{win,wsl}.json``):

=========================================  ========  ===================
quantity                                   bar       measured WIN / WSL
=========================================  ========  ===================
samples inside EE80, Collins lattice       < 3       1.4148 / 1.4148
samples inside EE80, co-moving lattice     > 30      102.47 / 102.00
|EE80(own lattice)/EE80(fine)-1|, Collins  > 0.08    0.17055 / 0.17055
|EE80(fine, Collins)/EE80(co-moving)-1|    < 0.03    0.00099 / 0.00361
=========================================  ========  ===================

The third row is a comparison of one transport with itself, so it is
bit-identical on the two builds; the whole cross-build spread of this fixture
lives in the fourth row's co-moving reference (46.1129 um WIN against 45.9016
um WSL, 0.46 %), which is why the bar there is 8x the worse reading and not
tighter.
"""

from __future__ import annotations

import warnings

import numpy as np

import lumenairy as la

LAM = 1.31e-6

#: The capstone's final leg lands 0.67 % of the way from the carrier's own
#: geometric focus.  ``A = 1 + z/R``; this fixture reproduces it.
A_NEAR_FOCUS = 0.0067
N, DX, W, R_IN = 256, 1e-6, 90e-6, -40e-3
Z = -R_IN * (1.0 - A_NEAR_FOCUS)


def _envelope():
    x = (np.arange(N) - N / 2) * DX
    xx, yy = np.meshgrid(x, x, indexing='xy')
    return np.exp(-(xx ** 2 + yy ** 2) / W ** 2).astype(np.complex128)


def _pitch(cr):
    return float(cr.dx if not isinstance(cr.dx, tuple) else cr.dx[0])


def _ee(intensity, dx, win, frac=0.8):
    """Encircled-energy radius, the capstone's own metric (``_ee_metrics``)."""
    n = intensity.shape[0]
    x = (np.arange(n) - n / 2) * dx
    xx, yy = np.meshgrid(x, x)
    jp, ip = np.unravel_index(np.argmax(intensity), intensity.shape)
    rad = np.sqrt((xx - x[ip]) ** 2 + (yy - x[jp]) ** 2)
    m = rad <= win
    iw, rw, pw = intensity[m], rad[m], intensity[m].sum()
    bins = np.linspace(0.0, win, 600)
    cum = np.array([iw[rw <= t].sum() for t in bins]) / pw
    return float(np.interp(frac, cum, bins))


def test_the_near_focus_ee80_gap_is_the_returned_lattice_not_the_field():
    """The p8 capstone's open near-focus question, SETTLED on a fast stand-in.

    Four claims, all measured in this process, none copied from a report:

    1. the leg is the right shape -- the Collins transport resolves a FLAT
       output reference here (``R_out`` is ``inf``, not the geometric
       ``R + z``) and runs its chirp-Z rather than falling back, so the pitch
       it returns is the floor ``2 r_out / N``;
    2. that pitch cannot MEASURE the spot -- fewer than 3 samples lie inside
       EE80, where the co-moving lattice puts more than 30;
    3. the reading is therefore a property of the LATTICE: driving the SAME
       transport onto the co-moving pitch with an explicit ``dx_out`` moves
       its own EE80 by more than 8 %;
    4. and on that common lattice the two transports agree -- better than
       3 % -- so the FIELD is not where they differ.

    Claim 3 is a transport compared with itself and carries no cross-build
    spread (measured 0.17055 on both builds).  Claim 4's spread is its
    co-moving reference's own (0.46 % between the builds), which is why its
    bar sits 8x above the worse reading.  Together they refute "the Collins
    field is wrong on this relay" and establish the remedy: a finer readout
    pitch, i.e. the caller naming ``dx_out`` or the floor being re-derived.
    """
    env = _envelope()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        co_moving = la.propagate_carrier_referenced(
            env, R_IN, Z, LAM, DX, transport='sziklas')
        collins = la.propagate_carrier_referenced(
            env, R_IN, Z, LAM, DX, transport='collins')
    dx_s, dx_c = _pitch(co_moving), _pitch(collins)

    # 1. the leg resolves a FLAT reference and stays on the chirp-Z.  A leg
    #    that had fallen back would return the geometric R + z and the
    #    co-moving pitch, which is exactly what this fixture must not be.
    assert np.isinf(collins.R), collins.R
    assert not np.isinf(co_moving.R), co_moving.R
    assert dx_c > 10.0 * dx_s, (dx_c, dx_s)
    assert any('under-sampled' in str(w.message) for w in caught), [
        str(w.message)[:90] for w in caught]

    win = min(0.45 * N * dx_s, 0.45 * N * dx_c)
    ee_s = _ee(np.abs(np.asarray(co_moving.env)) ** 2, dx_s, win)
    ee_c = _ee(np.abs(np.asarray(collins.env)) ** 2, dx_c, win)

    # 2. the returned lattice cannot measure the spot (1.41 vs 102 measured)
    assert ee_c / dx_c < 3.0, (ee_c, dx_c)
    assert ee_s / dx_s > 30.0, (ee_s, dx_s)

    # 3. the SAME transport on the co-moving pitch (0.17055, both builds)
    fine = la.propagate_carrier_referenced(
        env, R_IN, Z, LAM, DX, transport='collins', dx_out=dx_s,
        on_collins_sampling='ignore')
    ee_f = _ee(np.abs(np.asarray(fine.env)) ** 2, _pitch(fine), win)
    assert abs(ee_c / ee_f - 1.0) > 0.08, (ee_c, ee_f)

    # 4. ... and there the two transports agree (0.00099 WIN / 0.00361 WSL)
    assert abs(ee_f / ee_s - 1.0) < 0.03, (ee_f, ee_s)

"""
Lumenairy example 16 -- seeing what the pure 2-D solver is actually solving.

Demonstrates the geometry viewers added to
``lumenairy.elements.pmm.PMM2DStackPure`` (docs/audits/
AUDIT_PMM2D_PURE_VIEWER_2026_09_21.md):

  * ``stack.plot_geometry()`` -- one panel per layer, the exact ``(x, y)``
                                 walls the solver uses, no pixelation
  * ``stack.plot_section()``  -- a z cross-section sliced out of the same
                                 cell arrays, with slant drawn

The device is a metal-channel liquid-crystal cell of the kind used for a
reflective modulator: copper fingers of width ``W`` separated by channels of
width ``G``, the channel walls lined with alumina and then silicon nitride, the
liquid crystal filling what is left, all standing on a carbonitride base over a
copper mirror.  It is a good demonstration because it mixes everything the
viewers have to distinguish: a metal, three dielectrics, an ANISOTROPIC
liquid-crystal tensor, per-layer non-uniform walls, and a slanted sidewall.

The point of the example is the habit, not the device.  A cross-section drawn
by hand beside the code agrees with its author; one read out of the stack object
agrees with the physics.  Drawing this cell is how two silent approximations --
two conformal coats merged into one, and the wrong material under a 1 nm
adhesion layer -- were caught in the campaign this feature came from.

Run::

    python examples/16_pmm2d_pure_geometry_view.py

Writes ``examples/output/16_pmm2d_pure_geometry.png``.
"""
import math
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from lumenairy.elements.pmm import PMM2DStackPure, material_key
from lumenairy.elements.rcwa import uniaxial_tensor

NM = 1e-9
W, G, H = 240.0, 120.0, 325.0          # finger width, channel width, depth [nm]
T_AL, T_SIN, T_TA, T_BASE = 5.0, 15.0, 1.0, 40.0
WL = 1310 * NM

EPS_CU = complex(-83.131, 2.702)       # copper at 1310 nm
EPS_TA = complex(-147.785, 24.497)     # tantalum
EPS_AL2O3 = complex(1.746 ** 2, 0.0)
EPS_SI3N4 = complex(1.996 ** 2, 0.0)
EPS_BASE = complex(1.781 ** 2, 0.0)
N_O, N_E = 1.56, 1.90                  # the liquid crystal


def lc(phi_deg):
    """The crystal's permittivity tensor with the director at ``phi`` to x."""
    return np.asarray(uniaxial_tensor(N_O, N_E, math.pi / 2,
                                      phi=math.radians(phi_deg)))


def channel_cell(walls, inner_half, ring_half, inner, ring, outer):
    """One band of a two-channel cell: `inner` inside ``inner_half`` of each
    channel centreline, `ring` out to ``ring_half``, `outer` beyond."""
    edges = np.concatenate(([0.0], np.asarray(walls), [PERIOD]))
    centres = 0.5 * (edges[:-1] + edges[1:])
    n = len(centres)
    cell = np.zeros((n, n, 3, 3), dtype=complex)
    for i, x in enumerate(centres):
        for j, y in enumerate(centres):
            d = min(abs(x - c) for c in CHANNELS)
            e = min(abs(y - c) for c in CHANNELS)
            m = min(d, e)
            if m < inner_half:
                cell[i, j] = inner if inner.ndim == 2 else inner * np.eye(3)
            elif m < ring_half:
                cell[i, j] = ring * np.eye(3)
            else:
                cell[i, j] = outer * np.eye(3)
    return cell


PERIOD = 2.0 * (W + G)
CHANNELS = (0.5 * (W + G), 1.5 * (W + G))


def walls_at(*offsets):
    """Wall positions where each channel's material boundaries fall."""
    out = set()
    for d in offsets:
        for c in CHANNELS:
            out.add(round((c - (G / 2 - d)) % PERIOD, 6))
            out.add(round((c + (G / 2 - d)) % PERIOD, 6))
    return sorted(out)


def build(twist_deg=45.0, slant=0.035):
    """The coated channel cell, band by band, as the solver will hold it."""
    stack = PMM2DStackPure(PERIOD * NM, PERIOD * NM,
                           n_superstrate=N_O, n_substrate=np.sqrt(EPS_CU),
                           n_modes=5, n_orders=5, layer_grids="per-layer")
    tc = T_AL + T_SIN
    # EVERY patterned layer carries the SAME slant: the solver refuses a mixed
    # set, because the frame offset between two differently sheared nodal grids
    # is a real lateral translation.  A conformal coat on a slanted wall is
    # slanted too, so this is also what the device does.
    sl = (slant, 0.0)
    cap = np.array(walls_at(tc))
    body = np.array(walls_at(0.0, T_AL, tc))
    floor = np.array(walls_at(0.0))
    director = lc(twist_deg)
    # the conformal cap over the fingers, nitride outermost
    for t, eps in ((T_SIN, EPS_SI3N4), (T_AL, EPS_AL2O3)):
        stack.add_layer(t * NM, eps_cell=channel_cell(cap, G / 2 - tc, G / 2 - tc,
                                                      director, eps, eps),
                        x_walls=cap * NM, y_walls=cap * NM, slant=sl)
    # the finger body: Cu | Al2O3 | Si3N4 | liquid crystal | Si3N4 | Al2O3 | Cu
    edges = np.concatenate(([0.0], body, [PERIOD]))
    centres = 0.5 * (edges[:-1] + edges[1:])
    n = len(centres)
    cell = np.zeros((n, n, 3, 3), dtype=complex)
    for i, x in enumerate(centres):
        for j, y in enumerate(centres):
            m = min(min(abs(x - c) for c in CHANNELS),
                    min(abs(y - c) for c in CHANNELS))
            if m < G / 2 - tc:
                cell[i, j] = director
            elif m < G / 2 - T_AL:
                cell[i, j] = EPS_SI3N4 * np.eye(3)
            elif m < G / 2:
                cell[i, j] = EPS_AL2O3 * np.eye(3)
            else:
                cell[i, j] = EPS_CU * np.eye(3)
    stack.add_layer((H - tc) * NM, eps_cell=cell, x_walls=body * NM,
                    y_walls=body * NM, slant=sl)
    # the coated channel floor, then the adhesion layer under the fingers
    for t, eps in ((T_SIN, EPS_SI3N4), (T_AL, EPS_AL2O3)):
        stack.add_layer(t * NM, eps_cell=channel_cell(floor, G / 2, G / 2,
                                                      np.asarray(eps), eps, EPS_CU),
                        x_walls=floor * NM, y_walls=floor * NM, slant=sl)
    stack.add_layer(T_TA * NM,
                    eps_cell=channel_cell(floor, G / 2, G / 2,
                                          np.asarray(EPS_BASE), EPS_BASE, EPS_TA),
                    x_walls=floor * NM, y_walls=floor * NM, slant=sl)
    stack.add_layer(T_BASE * NM, eps=EPS_BASE)
    return stack


def main():
    stack = build()
    # a scalar may be named directly; a tensor is unhashable, so it is named
    # through material_key -- or pass the whole thing as (eps, name) pairs.
    names = {EPS_CU: "Cu", EPS_TA: "Ta", EPS_AL2O3: "Al2O3",
             EPS_SI3N4: "Si3N4", EPS_BASE: "SiCN",
             material_key(lc(45.0)): "LC, director 45 deg"}

    fig = plt.figure(figsize=(15.5, 8.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0])
    ax = fig.add_subplot(gs[0])
    stack.plot_section(ax=ax, material_names=names)
    ax.set_title("z cross-section, read out of the stack object", fontsize=10)
    sub = gs[1].subgridspec(1, len(stack._layers))
    stack.plot_geometry(axes=[fig.add_subplot(sub[i])
                              for i in range(len(stack._layers))],
                        material_names=names)
    fig.suptitle("PMM2DStackPure geometry views -- a coated metal-channel "
                 "liquid-crystal cell", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 0.97, 0.95))

    out = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "16_pmm2d_pure_geometry.png")
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")

    # The views needed no source at all: geometry is complete before
    # set_source, and drawing leaves the stack untouched.  Solving THIS cell is
    # a different matter -- its widest band carries 13 segments per axis and the
    # pure method's cost grows as the cube of that, so the full device belongs
    # on a cluster or on PMM2DStackHybrid.  Here we solve a deliberately coarse
    # twin, only to show that the object you drew is the object you solve.
    coarse = PMM2DStackPure(PERIOD * NM, PERIOD * NM, n_superstrate=N_O,
                            n_substrate=np.sqrt(EPS_CU), n_modes=4, n_orders=3,
                            layer_grids="per-layer")
    floor = np.array(walls_at(0.0))
    coarse.add_layer(H * NM,
                     eps_cell=channel_cell(floor, G / 2, G / 2,
                                           lc(45.0), EPS_SI3N4, EPS_CU),
                     x_walls=floor * NM, y_walls=floor * NM)
    coarse.add_layer(T_BASE * NM, eps=EPS_BASE)
    coarse.set_source(WL, theta=math.radians(12.0), phi=math.radians(30.0))
    orders, R, T, J = coarse.solve(jones=True)
    print(f"coarse twin: |J00| = {abs(np.asarray(J)[0, 0]):.4f}, "
          f"total R = {np.asarray(R)[0].sum():.4f}")


if __name__ == "__main__":
    main()

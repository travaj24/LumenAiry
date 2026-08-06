# Shared design-121 setup for the 2026-07-30 tilted-coarse-leg scoping study.
#
# LOCAL-ONLY: needs the 121 .zmx and the design-study runner (Schott Sellmeier
# coefficients).  Factored out of fan_multi_121.py VERBATIM so every script in
# this study starts from a bit-identical prescription split, DOE period, order
# table and chain-A (source -> DOE) field.  Nothing here is new physics; it is
# the same construction fan_multi_121.py performs inline.
#
# Chain A is CACHED to _chainA_<N>_<dx0>.npz because it costs ~1-2 min and is
# order-INDEPENDENT (the DOE decomposition happens after it).
import ast
import dataclasses
import os
import re
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore', message='.*prescription aperture.*')
warnings.filterwarnings('ignore', message='.*residual transverse.*')
warnings.filterwarnings('ignore', message='.*under-sampled.*')

_HERE = os.path.dirname(os.path.abspath(__file__))
# LOCAL-ONLY paths.  ``D121_ROOT`` overrides the Windows dev-box location, so
# the same runners drive from the WSL CI proxy as well -- niche C13 needed the
# design's OWN per-order table on the other BLAS build, and this one literal
# was the only thing that stopped it (2026-08-03).
_ROOT = os.environ.get(
    'D121_ROOT', r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics")
sys.path.insert(0, os.path.join(_ROOT, "Lumenairy"))
sys.path.insert(0, os.path.join(_ROOT, "Reverse_Symmetric_ASM"))

import lumenairy as la                                        # noqa: E402
from lumenairy import (GLASS_REGISTRY, GLASS_VALIDITY,        # noqa: E402
                       SELLMEIER_COEFFICIENTS)
from lumenairy.propagators.carrier import (                   # noqa: E402
    carrier_referenced_envelope)
from lumenairy.raytrace import Surface                        # noqa: E402
from lumenairy.raytrace.seidel import system_abcd_prescription  # noqa: E402
from lumenairy.raytrace.trace import surfaces_from_prescription  # noqa: E402

RUNNER = os.path.join(_ROOT, "Reverse_Symmetric_ASM",
                      "run_poc_119_120_v518.py")
ZMX = os.path.join(_ROOT, "Reverse_Symmetric_ASM", "tx4designstudy121",
                   "20260707 dll Tx02-MSOP16.zmx")
LAM = 1.31e-6
W0 = 4e-6
OBJ_GAP = 47.906284e-3
TRAILING = 7.7058e-3
FRAME_PITCH = 480e-6
DOE_ELEM = 6
ORDERS = (4, 8)
Z1 = 2e-3


def _register_glasses():
    src = open(RUNNER, 'r', encoding='utf-8', errors='ignore').read()
    m = re.search(r'_NEW_GLASSES\s*=\s*\{', src)
    i0 = m.end() - 1
    d = 0
    for i in range(i0, len(src)):
        if src[i] == '{':
            d += 1
        elif src[i] == '}':
            d -= 1
            if d == 0:
                break
    for g, (c, r, _nd) in ast.literal_eval(src[i0:i + 1]).items():
        if g not in SELLMEIER_COEFFICIENTS:
            SELLMEIER_COEFFICIENTS[g] = c
            GLASS_REGISTRY[g] = '__sellmeier__'
            GLASS_VALIDITY[g] = r


_register_glasses()
import copy                                                    # noqa: E402
import tx_design_study_sim as sim                              # noqa: E402


def geometry():
    """Return ``(groups_pre, groups_post, gap_to_doe, doe_period)``."""
    rx = la.load_zemax_zmx(ZMX)
    events = sim.group_elements_into_lenses(rx)
    rx_doe = copy.deepcopy(rx)
    for k in ('elements', 'all_thicknesses', 'surfaces', 'thicknesses'):
        rx_doe[k] = rx[k][DOE_ELEM:]
    b = float(system_abcd_prescription(rx_doe, LAM)[0][0, 1])
    period = LAM * abs(b) / FRAME_PITCH
    gapsum = 0.0
    first = True
    pre, post = [], []
    gap_to_doe = None
    for ev in events:
        if ev['type'] == 'doe' and gap_to_doe is None:
            gap_to_doe = ev['z_before'] + gapsum
            gapsum = 0.0
            continue
        if ev['type'] != 'lens':
            gapsum += ev['z_before']
            continue
        g = (OBJ_GAP - Z1) if first else (ev['z_before'] + gapsum)
        gapsum = 0.0
        (pre if gap_to_doe is None else post).append(
            {'prescription': ev['prescription'], 'gap_before': g})
        first = False
    return pre, post, gap_to_doe, period


def order_table(period, n_per=128):
    """Dammann order indices ``(mx, my)`` and complex amplitudes."""
    cache = os.path.join(_HERE, f'_dammann_121_{ORDERS[0]}x{ORDERS[1]}_'
                                f'{n_per}.npy')
    if os.path.exists(cache):
        nf = np.load(cache)
    else:
        nf, _f, _c = la.makedammann2d(
            periodx=period, periody=period, waveln=LAM,
            diforders=np.ones(ORDERS), phaselevels=8, phasesteps=4, itr=3000,
            seed=42, plot=False, cell_pixels=n_per)
        np.save(cache, nf)
    A = np.fft.fftshift(np.fft.fft2(nf)) / nf.size
    nx, ny = ORDERS
    cx = cy = n_per // 2
    P = np.abs(A) ** 2
    flat = np.argsort(P.ravel())[::-1][:nx * ny]
    oy, ox = np.unravel_index(flat, P.shape)
    mx, my = ox - cx, oy - cy
    o = np.lexsort((mx, my))
    mx, my = mx[o], my[o]
    return mx, my, A[my + cy, mx + cx]


def chain_a(n=1024, dx0=None, rs=4, nw=8, cache=True, final_leg='exact'):
    """Source envelope -> DOE plane.  Returns ``(env, R, dx, P_in)``.

    ``final_leg`` is part of the CACHE KEY.  It used to be hard-coded
    ``'paraxial'`` and absent from the filename, so switching the leg silently
    returned the previously-cached paraxial field and looked like a no-op.
    """
    dx0 = float(dx0 if dx0 is not None else 1.0e-6 * 2048 / n)
    fn = os.path.join(
        _HERE, f'_chainA_{n}_{dx0 * 1e9:.0f}nm_rs{rs}_{final_leg}.npz')
    if cache and os.path.exists(fn):
        d = np.load(fn)
        return d['env'], float(d['R']), float(d['dx']), float(d['P_in'])
    pre, _post, gap_to_doe, _per = geometry()
    zR = np.pi * W0 * W0 / LAM
    w_z1 = W0 * np.sqrt(1 + (Z1 / zR) ** 2)
    R1 = Z1 * (1 + (zR / Z1) ** 2)
    x = (np.arange(n) - n // 2) * dx0
    env0 = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
                  / w_z1 ** 2).astype(np.complex128)
    P_in = float(np.sum(np.abs(env0) ** 2)) * dx0 * dx0
    res = la.propagate_traced_carrier_chain(
        env0, pre, LAM, dx0, r_in=R1, ray_subsample=rs, n_workers=nw,
        final_distance=gap_to_doe, final_leg=final_leg)
    env = carrier_referenced_envelope(res.field, res.R, LAM, res.dx)
    if cache:
        np.savez_compressed(fn, env=env, R=float(res.R), dx=float(res.dx),
                            P_in=P_in)
    return env, float(res.R), float(res.dx), P_in


def post_surfaces(groups_post, trailing=TRAILING, stop_after=None,
                  back_off=0.0):
    """Sequential surface list for the post-DOE relay.

    ``stop_after`` (0-based group index) truncates after that group, and the
    trailing thickness becomes the following group's ``gap_before``.
    ``back_off`` pulls the final plane back by that many metres (used to place
    an exit REFERENCE plane short of the image plane)."""
    surfs = [Surface(radius=np.inf, conic=0.0, semi_diameter=np.inf,
                     glass_before='air', glass_after='air', is_mirror=False,
                     thickness=float(groups_post[0]['gap_before']),
                     label='doe_plane')]
    last = len(groups_post) - 1 if stop_after is None else int(stop_after)
    for j, g in enumerate(groups_post[:last + 1]):
        sf = surfaces_from_prescription(g['prescription'])
        if j < last:
            nxt = float(groups_post[j + 1]['gap_before'])
        else:
            nxt = (float(trailing) if stop_after is None
                   else float(groups_post[j + 1]['gap_before']))
            nxt -= float(back_off)      # ONLY the final leg is pulled back
        sf[-1] = dataclasses.replace(sf[-1], thickness=nxt)
        surfs.extend(sf)
    surfs.append(Surface(radius=np.inf, conic=0.0, semi_diameter=np.inf,
                         glass_before='air', glass_after='air',
                         is_mirror=False, thickness=0.0, label='img'))
    return surfs

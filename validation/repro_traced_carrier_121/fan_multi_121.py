# Design-121 32-ORDER DOE FAN through propagate_traced_carrier_chain_multi
# (niche D2 / roadmap ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27 P1b).
#
# LOCAL-ONLY: needs the 121 .zmx and the design-study runner (for the Schott
# Sellmeier coefficients).  This is the P1 ACCEPTANCE runner: does the 8x4
# order fan reconstruct with per-frame power matching the DAMMANN DESIGN's own
# uniformity, instead of the 0.47 +/- 0.51 %/frame scramble measured when the
# fan was pushed through the chain MULTIPLEXED at v5.28?
#
# Construction (the consumer-side split P2 will eventually remove):
#   1. chain A -- source envelope -> the DOE plane (the groups before the DOE,
#      with the last-lens -> DOE gap as final_distance).  No readout.
#   2. the DOE is decomposed into its ORDERS: the Dammann cell's own DFT gives
#      the complex amplitude a(m,n) of order (m,n), whose tilt is
#      (L, M) = (m, n) * lambda / period.  Many periods illuminate the beam,
#      so this order decomposition is the DOE's exact action.
#   3. chain B -- the orchestrator carries each order as its OWN congruence
#      (TiltedCarrier(R_doe, L, M), weight a(m,n)) through the remaining
#      groups + the trailing leg to the MSoP plane, and recombines on one
#      common image grid with a per-frame readout tile.
#
# Env knobs: RN (input grid N), RS (ray_subsample), NOUT/DXO (common grid),
# TILE (per-frame readout window), NW (Newton workers), ORDERS_ONLY=1 to stop
# after the design-side order table.
#   CW    -- congruence_workers (niche D8): K independent orders in a spawn
#            pool.  Default 1 (serial).  Guards and accumulation stay in the
#            parent in ascending k, so the answer is FP-identical to serial.
#   RAMB  -- the fine grid's RAM-budget INTENT: 'auto' (the box's own
#            get_ram_budget(), which each congruence worker sees DIVIDED by
#            CW), 'inf' (clamp off), or a number of GB.
#
# THE GRID OF RECORD IS ASSERTED, NOT ASSUMED.  ADJUDICATION_NFC_8192 sec 2.1:
# on a 137 GB box this runner's own NFC=16384 default silently ran at 8192 --
# the RAM clamp degraded it, said so in a warning, and the acceptance banner
# passed anyway.  ``_grid_intent.preflight`` now PROVES before chain B that the
# clamp cannot bind and that the box can hold CW workers, and refuses (exit 2)
# if it cannot; ``assert_no_grid_degradation`` re-checks it afterwards against
# the warnings the run actually raised, the workers' included.  Exit 2 = this
# box cannot do what you asked; exit 3 = the pre-flight said it could and it
# did not.
#
# IMPORT-SAFE: everything below the constants is behind
# ``if __name__ == '__main__':``.  Under spawn every congruence worker
# re-imports this module, and an unguarded body would run the whole acceptance
# again in each child -- which is why CW could not be used at all before
# (AUDIT_TRACED_SPEED_2026_08_09 sec 3.2).
#
# ACCEPTANCE METRIC = per-frame power OF THE RECOMBINED FIELD, measured by
# partitioning ``res.field`` on the 480 um frame lattice about centres from an
# EXACT SKEW RAY TRACE through the post-DOE surfaces (not the library's own
# paraxial chief-ray predictor, so the partition does not inherit the
# placement it is checking).  That is what the roadmap's acceptance is about:
# "the 32-order fan RECONSTRUCTS with per-frame power matching the Dammann
# design uniformity".
#
# It is NOT ``power_exit``.  An adversarial pass killed that as the headline:
# ``power_exit`` is read per congruence from its own chain stages BEFORE
# recombination, tiling or readout, so it equals |a(m,n)|^2 * P_doe *
# throughput_k by construction -- it echoes the Dammann design and cannot see
# a recombination failure at all.  Measured on the reproduced v5.28 scramble
# (readout_tile=None + on_replica='ignore'), the recombined field's per-frame
# shares were 16.6/24.7/24.2/34.5 % against a design 25.0/25.0/25.0/25.0 %,
# while the power_exit metric read max |share/design - 1| = 0.00092 -- a clean
# pass on a fully scrambled reconstruction.
#
# ``power_exit`` / ``throughput`` (power_exit/power_in) are still printed:
# they are the tile-INDEPENDENT vignetting, and keeping them apart from the
# window numbers is what stopped a readout-tile clipping artefact being
# reported as field-angle-dependent vignetting.  ``capture``
# (power_out/power_exit) is the window's share and should read ~1.
import ast, dataclasses, os, re, sys, time, warnings
import numpy as np

# NEVER blanket-silence here.  The first cut of this runner did
# (`warnings.filterwarnings('ignore')`), which hid the only signal that the
# readout window was in the periodic-replica regime -- the exact class of
# silent wrong answer this acceptance is supposed to detect.  Silence only the
# two known-noisy, understood categories and let everything else through; the
# replica / clipping guards are orchestrator-owned errors now, not warnings.
warnings.filterwarnings('ignore', message='.*prescription aperture.*')
warnings.filterwarnings('ignore', message='.*residual transverse.*')
warnings.filterwarnings('ignore', message='.*under-sampled.*')
sys.path.insert(0, r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy")
sys.path.insert(0, r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Reverse_Symmetric_ASM")
import lumenairy as la
from lumenairy import GLASS_REGISTRY, GLASS_VALIDITY, SELLMEIER_COEFFICIENTS
from lumenairy.propagators.carrier import carrier_referenced_envelope
from lumenairy.raytrace import Surface, make_ray, trace
from lumenairy.raytrace.seidel import system_abcd_prescription
from lumenairy.raytrace.trace import surfaces_from_prescription

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _grid_intent as gi        # noqa: E402  (import-safe: no module body work)

RUNNER = (r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics"
          r"\Reverse_Symmetric_ASM\run_poc_119_120_v518.py")
ZMX = (r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics"
       r"\Reverse_Symmetric_ASM\tx4designstudy121\20260707 dll Tx02-MSOP16.zmx")
lam = 1.31e-6
w0 = 4e-6
OBJ_GAP = 47.906284e-3
TRAILING = 7.7058e-3
FRAME_PITCH = 480e-6
DOE_ELEM = 6
ORDERS = (4, 8)          # (nx, ny) as the runner sets sim.DAMMANN_ORDERS


# --- IMPORT-SAFE BOUNDARY --------------------------------------------------
# Under spawn a congruence worker RE-IMPORTS this module (as __mp_main__), so
# everything with a side effect -- the glass registration, the .zmx load, the
# Dammann solve, both chains -- has to sit behind this guard or it runs again
# in every child.  The library detects an unguarded caller and refuses the
# pool with a message naming this fix (carrier.py _multi_looks_like_spawn_
# bootstrap; _lens_traced._script_has_main_guard forces the Newton pool
# serial for the same reason), and that backstop stays -- this guard is what
# makes CW>1 REACHABLE at all.
if __name__ == '__main__':
    # --- Schott glasses the 121 design uses, parsed out of the runner ----------
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
    for g, (c, r, nd) in ast.literal_eval(src[i0:i + 1]).items():
        if g not in SELLMEIER_COEFFICIENTS:
            SELLMEIER_COEFFICIENTS[g] = c
            GLASS_REGISTRY[g] = '__sellmeier__'
            GLASS_VALIDITY[g] = r

    import copy

    import tx_design_study_sim as sim

    rx = la.load_zemax_zmx(ZMX)
    events = sim.group_elements_into_lenses(rx)

    # B(DOE -> MSoP) sets the DOE period for a 480 um frame pitch (the runner's
    # own derivation, reproduced here so this script is self-contained).
    rx_doe = copy.deepcopy(rx)
    for k in ('elements', 'all_thicknesses', 'surfaces', 'thicknesses'):
        rx_doe[k] = rx[k][DOE_ELEM:]
    B_DOE_TO_MS = float(system_abcd_prescription(rx_doe, lam)[0][0, 1])
    DOE_PERIOD = lam * abs(B_DOE_TO_MS) / FRAME_PITCH
    print(f"B(DOE->MS) = {B_DOE_TO_MS * 1e3:+.4f} mm  ->  DOE period "
          f"{DOE_PERIOD * 1e6:.3f} um, frame pitch {FRAME_PITCH * 1e6:.0f} um")

    # --- split the group list at the DOE ---------------------------------------
    z1 = 2e-3
    gapsum = 0.0
    first = True
    groups_pre, groups_post = [], []
    gap_to_doe = None          # last-lens exit -> DOE plane
    for ev in events:
        if ev['type'] == 'doe' and gap_to_doe is None:
            gap_to_doe = ev['z_before'] + gapsum
            gapsum = 0.0
            continue
        if ev['type'] != 'lens':
            gapsum += ev['z_before']
            continue
        g = (OBJ_GAP - z1) if first else (ev['z_before'] + gapsum)
        gapsum = 0.0
        (groups_pre if gap_to_doe is None else groups_post).append(
            {'prescription': ev['prescription'], 'gap_before': g})
        first = False
    print(f"groups: {len(groups_pre)} before the DOE, {len(groups_post)} after; "
          f"last-lens -> DOE gap {gap_to_doe * 1e3:.4f} mm, "
          f"DOE -> next-lens gap {groups_post[0]['gap_before'] * 1e3:.4f} mm")

    # --- the Dammann design's own order table ----------------------------------
    N_PER = int(os.environ.get('DPX', '128'))       # cell pixels per period
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f'_dammann_121_{ORDERS[0]}x{ORDERS[1]}_{N_PER}.npy')
    if os.path.exists(cache):
        nf = np.load(cache)
    else:
        nf, _ff, _cd = la.makedammann2d(
            periodx=DOE_PERIOD, periody=DOE_PERIOD, waveln=lam,
            diforders=np.ones(ORDERS), phaselevels=8, phasesteps=4, itr=3000,
            seed=42, plot=False, cell_pixels=N_PER)
        np.save(cache, nf)
    print(f"Dammann cell {nf.shape}, |amp| range "
          f"{np.abs(nf).min():.3f}-{np.abs(nf).max():.3f}")

    # order amplitudes = the DFT of one period (unitary, so sum|a|^2 <= 1)
    A = np.fft.fftshift(np.fft.fft2(nf)) / nf.size
    nx, ny = ORDERS
    cx = cy = N_PER // 2
    # Dammann orders for an even count sit at the HALF-integer lattice
    # +-1/2, +-3/2, ... in units of 1/period; the IFTA embeds them on the
    # even/odd sublattice of the doubled cell.  Locate them by peak search
    # instead of assuming, then verify the count.
    P = np.abs(A) ** 2
    flat = np.argsort(P.ravel())[::-1][:nx * ny]
    oy, ox = np.unravel_index(flat, P.shape)
    mx = ox - cx
    my = oy - cy
    order = np.lexsort((mx, my))
    mx, my = mx[order], my[order]
    amps = A[my + cy, mx + cx]
    # KEEP="m,n;m,n;..." restricts the fan to a SUBSET of its own orders.  Its
    # purpose is the acceptance instrument's own fail-before/pass-after: a 2x2
    # block of adjacent orders is cheap enough (a 4096-square common grid instead
    # of 16384) to run BOTH the replica-scrambled readout (TILE=none) and the
    # guarded one, and see the verdict below flip.  Shares are re-normalised over
    # the kept orders, so the design column stays self-consistent.
    _keep = os.environ.get('KEEP')
    if _keep:
        want = {tuple(int(v) for v in pair.split(','))
                for pair in _keep.split(';') if pair.strip()}
        sel = [i for i in range(len(mx)) if (int(mx[i]), int(my[i])) in want]
        if len(sel) != len(want):
            raise SystemExit(f"KEEP: asked for {sorted(want)}, found "
                             f"{sorted((int(mx[i]), int(my[i])) for i in sel)}")
        mx, my, amps = mx[sel], my[sel], amps[sel]
        print(f"KEEP: {len(sel)} of {nx * ny} orders -> "
              f"{[(int(a), int(b)) for a, b in zip(mx, my)]}")
    p_ord = np.abs(amps) ** 2
    eff = float(p_ord.sum())
    uni_design = float(p_ord.min() / p_ord.max())
    print(f"design orders (m,n) x: {sorted(set(mx.tolist()))}  "
          f"y: {sorted(set(my.tolist()))}")
    print(f"DESIGN: {len(amps)} orders, efficiency {eff * 100:.2f}%, "
          f"per-frame {p_ord.mean() * 100:.3f} +/- {p_ord.std() * 100:.3f} % "
          f"(of the incident), uniformity min/max {uni_design:.4f}")
    print(f"        per-frame OF THE DIFFRACTED power: "
          f"{(p_ord / eff).mean() * 100:.3f} +/- {(p_ord / eff).std() * 100:.3f} %")
    if os.environ.get('ORDERS_ONLY') == '1':
        raise SystemExit(0)

    # --- chain A: source -> the DOE plane ---------------------------------------
    N = int(os.environ.get('RN', '1024'))
    RS = int(os.environ.get('RS', '4'))
    NW = int(os.environ.get('NW', '8'))
    dx0 = float(os.environ.get('DX0', str(1.0e-6 * 2048 / N)))
    zR = np.pi * w0 * w0 / lam
    w_z1 = w0 * np.sqrt(1 + (z1 / zR) ** 2)
    R1 = z1 * (1 + (zR / z1) ** 2)
    x = (np.arange(N) - N // 2) * dx0
    env0 = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / w_z1 ** 2
                  ).astype(np.complex128)
    P_in = float(np.sum(np.abs(env0) ** 2)) * dx0 * dx0
    print(f"\nconfig: N={N} dx0={dx0 * 1e6:.4f} um rs={RS} nw={NW}")

    t0 = time.time()
    pre = la.propagate_traced_carrier_chain(
        env0, groups_pre, lam, dx0, r_in=R1, ray_subsample=RS, n_workers=NW,
        final_distance=gap_to_doe)
    env_doe = carrier_referenced_envelope(pre.field, pre.R, lam, pre.dx)
    P_doe = float(np.sum(np.abs(env_doe) ** 2)) * pre.dx * pre.dx
    print(f"chain A done {time.time() - t0:.0f}s: R_doe = {pre.R * 1e3:.4f} mm, "
          f"dx = {pre.dx * 1e6:.4f} um, power {P_doe / P_in * 100:.2f}% of input")

    # --- chain B: one congruence per order --------------------------------------
    DXO = float(os.environ.get('DXO', '0.2e-6'))
    # TILE default 1024: at DXO=0.4 um that is a 409.6 um window, inside the
    # measured 483-492 um Bluestein period of this readout (so no periodic
    # replicas -- the orchestrator refuses those anyway) and wide enough that the
    # measured capture is 1.00000, i.e. the window is NOT clipping the halo.  The
    # earlier 512 default clipped a field-angle-dependent ~1-2%, which is what got
    # mis-reported as vignetting.
    # ``TILE=auto`` routes to readout_tile='auto', which sizes the window from the
    # SHORTEST Bluestein period over all 32 orders (measured in a 16-px probe
    # pass, so no big readout and no order dependence).  It is exercised here and
    # lands on the same physics; the explicit default keeps the acceptance run at
    # K chain runs instead of 'auto''s 2K.
    # TILE=none is the historical full-grid readout with BOTH guards opted out --
    # i.e. the v5.28 replica scramble on purpose.  It exists so the acceptance
    # below can be shown to FAIL on a scrambled reconstruction (the metric it
    # replaced, per-congruence power_exit, printed a clean pass on exactly that).
    TILE = os.environ.get('TILE', '1024')
    TILE = TILE if TILE in ('auto', 'none') else int(TILE)
    GUARD_KW = (dict(readout_tile=None, on_replica='ignore',
                     on_readout_clip='ignore') if TILE == 'none'
                else dict(readout_tile=TILE))
    span = (max(abs(mx).max(), abs(my).max()) + 1.0) * 2.0 * FRAME_PITCH
    NOUT = int(os.environ.get('NOUT', str(
        int(2 ** np.ceil(np.log2(span / DXO))))))
    print(f"common grid: dx_out={DXO * 1e6:.3f} um N_out={NOUT} "
          f"({NOUT * DXO * 1e3:.3f} mm), tile {TILE}"
          + (f" ({TILE * DXO * 1e6:.1f} um)"
             if TILE not in ('auto', 'none') else ""))

    # --- frame centres from an EXACT SKEW RAY TRACE (independent of the library's
    # --- paraxial chief-ray predictor, which is what places the readout tiles) --
    def _post_surfaces():
        surfs = [Surface(radius=np.inf, conic=0.0, semi_diameter=np.inf,
                         glass_before='air', glass_after='air', is_mirror=False,
                         thickness=float(groups_post[0]['gap_before']),
                         label='doe_plane')]
        for j, g in enumerate(groups_post):
            sf = surfaces_from_prescription(g['prescription'])
            nxt = (float(groups_post[j + 1]['gap_before'])
                   if j + 1 < len(groups_post) else float(TRAILING))
            sf[-1] = dataclasses.replace(sf[-1], thickness=nxt)
            surfs.extend(sf)
        surfs.append(Surface(radius=np.inf, conic=0.0, semi_diameter=np.inf,
                             glass_before='air', glass_after='air',
                             is_mirror=False, thickness=0.0, label='img'))
        return surfs


    _SURFS = _post_surfaces()
    tilts = [(float(m_) * lam / DOE_PERIOD, float(n_) * lam / DOE_PERIOD)
             for m_, n_ in zip(mx, my)]
    exact_c = np.array([
        [float(trace(make_ray(0.0, 0.0, L, M, wavelength=lam), _SURFS,
                     lam).image_rays.x[0]),
         float(trace(make_ray(0.0, 0.0, L, M, wavelength=lam), _SURFS,
                     lam).image_rays.y[0])] for L, M in tilts])
    # cluster the x centres to the nearest micron before differencing: two
    # orders in different rows land on the same column to ~0.1 um
    _pitch = np.diff(np.unique(np.round(exact_c[:, 0] * 1e6))) * 1e-6
    print(f"exact skew-trace frame centres: x "
          f"{exact_c[:, 0].min() * 1e3:+.4f}..{exact_c[:, 0].max() * 1e3:+.4f} mm,"
          f" y {exact_c[:, 1].min() * 1e3:+.4f}..{exact_c[:, 1].max() * 1e3:+.4f}"
          f" mm; implied x-pitch {_pitch.mean() * 1e6:.3f} +/- "
          f"{_pitch.std() * 1e6:.3f} um (design {FRAME_PITCH * 1e6:.0f})")

    congruences = []
    for m_, n_, a in zip(mx, my, amps):
        congruences.append({
            'field': env_doe,
            'name': f'({m_:+d},{n_:+d})',
            'weight': complex(a),
            'carrier': la.TiltedCarrier(float(pre.R),
                                        float(m_) * lam / DOE_PERIOD,
                                        float(n_) * lam / DOE_PERIOD)})

    # LEG selects the final leg (niche D6).  'auto' resolves to the EXACT
    # high-NA final leg at 121's na_exit = 0.405 and has been this runner's
    # default since the 2026-08-06 capstone: the paraxial-era extreme order read
    # EE3 65.3% where exact reads 90.1% against the skew-ray+Debye oracle
    # (docs/audits/CAPSTONE_D121_2026_08_06.md).  'paraxial' remains available
    # by env for the historical comparison ONLY -- it is not a valid acceptance
    # configuration.  The exact leg carries a tilted congruence at the cost of
    # an axis-centred retrace window that has to hold the optical axis AND the
    # chief-ray-displaced beam (measured 1.48x wider on the extreme 121 order),
    # so NFC has to grow with it.
    LEG = os.environ.get('LEG', 'auto')
    NFC = int(os.environ.get('NFC', '16384'))
    WF = float(os.environ.get('WF', '4.0'))
    # CW = congruence_workers (niche D8).  Available only because this module
    # is import-safe (see the guard above): under spawn the pool re-imports
    # __main__, and the library refuses CW>1 from an unguarded caller.
    CW = int(os.environ.get('CW', '1'))
    # RAMB = the fine grid's RAM-budget INTENT.  Stated, proven, and asserted
    # afterwards -- never left to the clamp to decide quietly.  NOTE the box
    # budget a congruence worker sees is get_ram_budget() // CW, which is why
    # CW is an argument to the pre-flight.
    RAMB = os.environ.get('RAMB', 'auto')
    _RB = gi.preflight(NFC, RAMB, workers=CW, n_px=int(pre.field.size),
                       n_out=NOUT, paraxial=(LEG == 'paraxial'),
                       label=f"design-121 fan, {len(congruences)} orders, "
                             f"NFC={NFC} CW={CW}")
    _og = dict(dx_out=DXO, N_out=NOUT)
    if LEG != 'paraxial':
        _og.update(n_fine_cap=NFC, window_factor=WF)
    if _RB is not None:
        _og['ram_budget'] = _RB
    t0 = time.time()
    with gi.record_warnings() as _wrec:
        res = la.propagate_traced_carrier_chain_multi(
            congruences, groups_post, lam, pre.dx,
            output_grid=_og, **GUARD_KW,
            final_distance=TRAILING, ray_subsample=RS, n_workers=NW,
            final_leg=LEG, on_mem_budget='warn',
            congruence_workers=(CW if CW > 1 else None),
            on_tilt_exact_grid=os.environ.get('OTEG', 'error'),
            progress=lambda k, K, nm: print(f"  [{k + 1}/{K}] {nm}",
                                            flush=True))
    print(f"chain B done {time.time() - t0:.0f}s  (final_leg={LEG!r}"
          + (f", n_fine_cap={NFC}, window_factor={WF}" if LEG != 'paraxial'
             else "") + f", congruence_workers={CW})")
    if LEG != 'paraxial':
        gi.assert_no_grid_degradation(_wrec, NFC, label="design-121 fan")

    # --- per-frame power --------------------------------------------------------
    # power_out is the READOUT-WINDOW power (tile-dependent); throughput /
    # power_exit are measured at the chain exit and are tile-INDEPENDENT.  Keeping
    # them apart is the point: the first cut of this study reported the tile's
    # field-angle-dependent clipping as physical vignetting.
    pf = np.array([c['power_out'] for c in res.congruences])
    pin = np.array([c['power_in'] for c in res.congruences])
    pex = np.array([c['power_exit'] for c in res.congruences])
    cap = np.array([c['capture'] for c in res.congruences])
    thr = np.array([c['throughput'] for c in res.congruences])
    per = np.array([c['readout_period'] for c in res.congruences])
    clip = np.array([c['clipped'] for c in res.congruences])
    frac_in = pf / P_in                       # of the incident beam
    frac_dif = pf / pf.sum()                  # of the delivered fan
    ex_in = pex / P_in
    ex_dif = pex / pex.sum()
    share_des = p_ord / eff

    # ===========================================================================
    # THE ACCEPTANCE: per-frame power OF THE RECOMBINED FIELD
    # ===========================================================================
    # Partition res.field into 480 um cells about the EXACT ray-traced frame
    # centres and integrate.  This is the only metric here that can see a
    # recombination failure: power_exit is read from each congruence's own chain
    # stages BEFORE recombination, so it reproduces the Dammann design by
    # construction (verified: on the reproduced v5.28 scramble it read
    # max |share/design - 1| = 0.00092 while the field's shares were
    # 16.6/24.7/24.2/34.5 % against a 25/25/25/25 design).
    F = np.asarray(res.field)
    Ifield = np.abs(F) ** 2
    P_tot = float(Ifield.sum()) * DXO * DXO
    halfpx = int(round(0.5 * FRAME_PITCH / DXO))
    cellP = np.zeros(len(mx))
    for k, (xc, yc) in enumerate(exact_c):
        ic = int(round(xc / DXO + NOUT / 2))
        jc = int(round(yc / DXO + NOUT / 2))
        i0_, i1_ = max(ic - halfpx, 0), min(ic + halfpx, NOUT)
        j0_, j1_ = max(jc - halfpx, 0), min(jc + halfpx, NOUT)
        cellP[k] = float(Ifield[j0_:j1_, i0_:i1_].sum()) * DXO * DXO
    cell_in = cellP / P_in
    cell_share = cellP / cellP.sum()
    cell_uni = cell_in.min() / cell_in.max()
    cell_ratio = np.abs(cell_share / share_des - 1).max()
    cell_corr = float(np.corrcoef(share_des, cell_share)[0, 1])
    print()
    print("PER-FRAME POWER OF THE RECOMBINED FIELD -- ACCEPTANCE")
    print(f"  cells cover      : sum(cells)/total field power = "
          f"{cellP.sum() / P_tot:.9f}   (total field "
          f"{P_tot / P_in * 100:.4f} % of incident)")
    print(f"  of the incident  : {cell_in.mean() * 100:.4f} +/- "
          f"{cell_in.std() * 100:.4f} %   uniformity {cell_uni:.4f}")
    print(f"  design           : {p_ord.mean() * 100:.4f} +/- "
          f"{p_ord.std() * 100:.4f} %   uniformity {uni_design:.4f}")
    print(f"  share of fan     : {cell_share.mean() * 100:.4f} +/- "
          f"{cell_share.std() * 100:.4f} %   (design "
          f"{share_des.mean() * 100:.4f} +/- {share_des.std() * 100:.4f})")
    cell_rms = float(np.sqrt(np.mean((cell_share / share_des - 1.0) ** 2)))
    print(f"  max |share/design - 1| = {cell_ratio:.5f}   rms {cell_rms:.5f}")
    print(f"  design->measured correlation {cell_corr:.6f}   (DIAGNOSTIC only:")
    print("     over few frames whose design shares are near-equal this is")
    print("     noise-dominated -- 0.83 on a clean 2x2 block, 0.99 on the full")
    print("     32-order fan, -0.65 on the replica scramble.  The verdict uses")
    print("     the ratio/uniformity/coverage/power checks, which separate the")
    print("     scramble from the design by 100x rather than by a correlation.)")
    print(f"  cell power vs the library's power_out: max |ratio - 1| = "
          f"{np.abs(cellP / pf - 1).max():.3e}")
    lib_c = np.array([c['chief_ray'] for c in res.congruences])
    print(f"  library chief_ray vs exact skew trace: mean "
          f"{np.hypot(*(lib_c - exact_c).T).mean() * 1e6:.3f} um  max "
          f"{np.hypot(*(lib_c - exact_c).T).max() * 1e6:.3f} um")
    _checks = [
        ('cells cover the field (sum/total = 1)',
         abs(cellP.sum() / P_tot - 1.0) < 1e-3),
        ('total field power <= incident (no power created)',
         P_tot <= P_in * (1.0 + 1e-9)),
        ('per-frame uniformity >= 0.99 x design', cell_uni >= 0.99 * uni_design),
        ('max |share/design - 1| < 0.02', cell_ratio < 0.02),
        ('rms |share/design - 1| < 0.01', cell_rms < 0.01),
    ]
    print("  VERDICT: " + ("PASS" if all(v for _n, v in _checks) else "FAIL"))
    for _n, v in _checks:
        print(f"    [{'ok' if v else 'FAIL'}] {_n}")

    print()
    print("PER-FRAME DELIVERED POWER (tile-INDEPENDENT, power_exit) -- DIAGNOSTIC")
    print("  NOT the acceptance: read before recombination, so it echoes the")
    print("  design by construction and cannot see a scrambled reconstruction.")
    print(f"  of the incident : {ex_in.mean() * 100:.3f} +/- "
          f"{ex_in.std() * 100:.3f} %   uniformity "
          f"{ex_in.min() / ex_in.max():.4f}")
    print(f"  share of fan    : {ex_dif.mean() * 100:.4f} +/- "
          f"{ex_dif.std() * 100:.4f} %   max |share/design - 1| = "
          f"{np.abs(ex_dif / share_des - 1).max():.5f}")
    print()
    print("PER-FRAME WINDOW POWER (tile-DEPENDENT, power_out) -- for continuity")
    print("PER-FRAME POWER (of the incident beam):")
    print(f"  measured  {frac_in.mean() * 100:.3f} +/- {frac_in.std() * 100:.3f} %"
          f"   min/max uniformity {frac_in.min() / frac_in.max():.4f}")
    print(f"  design    {p_ord.mean() * 100:.3f} +/- {p_ord.std() * 100:.3f} %"
          f"   min/max uniformity {uni_design:.4f}")
    print("PER-FRAME SHARE (of the delivered fan):")
    print(f"  measured  {frac_dif.mean() * 100:.3f} +/- {frac_dif.std() * 100:.3f} %"
          f"   min/max {frac_dif.min() / frac_dif.max():.4f}")
    p_des = p_ord / p_ord.sum()
    print(f"  design    {p_des.mean() * 100:.3f} +/- {p_des.std() * 100:.3f} %"
          f"   min/max {p_des.min() / p_des.max():.4f}")
    print(f"  design->measured share correlation "
          f"{np.corrcoef(p_des, frac_dif)[0, 1]:.5f}; "
          f"max |share ratio - 1| = {np.abs(frac_dif / p_des - 1).max():.4f}")
    print(f"CHAIN THROUGHPUT (tile-INDEPENDENT, power_exit/power_in): "
          f"{pex.sum() / pin.sum():.5f}  per-order "
          f"{thr.min():.5f}..{thr.max():.5f} (spread {thr.max() - thr.min():.2e})")
    print(f"READOUT CAPTURE (tile-DEPENDENT, power_out/power_exit): "
          f"{cap.mean():.5f}  per-order {cap.min():.5f}..{cap.max():.5f} "
          f"(spread {cap.max() - cap.min():.2e})")
    print(f"  readout Bluestein period {per.min() * 1e6:.1f}..{per.max() * 1e6:.1f}"
          f" um -> largest safe tile "
          f"{int(2 * (per.min() / DXO // 2))} px; tile used "
          f"{res.congruences[0]['tile']} px "
          f"({res.congruences[0]['tile'] * DXO * 1e6:.1f} um)")
    print(f"  max clip {clip.max():.3g}")

    # --- per-frame SPOT quality (measured on the recombined field) --------------
    # NB at LEG='paraxial' (this runner's default) these are expected to be WORSE
    # than the single-beam acceptance's exact-leg 3.450 um / 88.8 / 99.6 -- the
    # paraxial readout is ~200 rad of wavefront wrong at this exit NA.  Since
    # niche D6 the tilted congruence CAN take the exact leg: run LEG=auto (with
    # NFC >= 16384 on design 121) and the same numbers improve materially.
    # Reported either way so the cost is visible rather than assumed.
    fwhms, ee3s, ee6s, ee12s, cent = [], [], [], [], []
    for c in res.congruences:
        NT = int(c['tile'])
        r0, c0 = c['tile_origin']
        tile = F[max(r0, 0):r0 + NT, max(c0, 0):c0 + NT]
        I = np.abs(tile) ** 2
        tot = float(I.sum())
        iy, ix = np.unravel_index(np.argmax(I), I.shape)
        rr = np.hypot((np.arange(I.shape[1]) - ix)[None, :] * DXO,
                      (np.arange(I.shape[0]) - iy)[:, None] * DXO)
        nb = min(NT // 2, 400)
        ring = np.clip((rr / DXO).astype(int), 0, nb)
        s = np.bincount(ring.ravel(), weights=I.ravel(), minlength=nb + 1)
        cn = np.bincount(ring.ravel(), minlength=nb + 1)
        prof = (s[:nb] / np.maximum(cn[:nb], 1)) / I[iy, ix]
        rb = (np.arange(nb) + 0.5) * DXO
        idx = np.where(prof < 0.5)[0]
        fwhms.append(2 * rb[idx[0]] if len(idx) else np.nan)
        ee3s.append(float(I[rr <= 3e-6].sum()) / tot)
        ee6s.append(float(I[rr <= 6e-6].sum()) / tot)
        ee12s.append(float(I[rr <= 12e-6].sum()) / tot)
        axx = (np.arange(I.shape[1]) - NT / 2) * DXO
        axy = (np.arange(I.shape[0]) - NT / 2) * DXO
        cent.append((c['tile_centre'][0] + float((I.sum(axis=0) * axx).sum() / tot)
                     - c['chief_ray'][0],
                     c['tile_centre'][1] + float((I.sum(axis=1) * axy).sum() / tot)
                     - c['chief_ray'][1]))
    fwhms, ee3s, ee6s = np.array(fwhms), np.array(ee3s), np.array(ee6s)
    ee12s = np.array(ee12s)
    cerr = np.hypot(*np.array(cent).T)
    print()
    print(f"PER-FRAME SPOT (final_leg={LEG!r}, in-frame EE): "
          f"FWHM {fwhms.mean() * 1e6:.3f} +/- {fwhms.std() * 1e6:.3f} um, "
          f"EE3 {ee3s.mean() * 100:.1f} +/- {ee3s.std() * 100:.1f} %, "
          f"EE6 {ee6s.mean() * 100:.1f} +/- {ee6s.std() * 100:.1f} %, "
          f"EE12 {ee12s.mean() * 100:.1f} +/- {ee12s.std() * 100:.1f} %")
    print(f"  centroid vs predicted chief ray: "
          f"{cerr.mean() * 1e6:.3f} +/- {cerr.std() * 1e6:.3f} um "
          f"(max {cerr.max() * 1e6:.3f} um)")
    print("  !! THE PER-FRAME SPOT NUMBERS ABOVE ARE A LOWER BOUND ON THE DESIGN,")
    print("     NOT THE DESIGN'S PERFORMANCE.  An independent skew-ray + Debye")
    print("     oracle of the same .zmx says design 121 is EQUALLY diffraction-")
    print("     limited at every order: EE3 ~90.7 %, EE6 ~99.9 %, rms wavefront")
    print("     0.078-0.082 waves.  Measured on the exact leg (NFC 12288, WF 4.0,")
    print("     dx_out 0.4 um) the chain delivers EE3 87.6 / 86.0 / 68.1 / 65.3 %")
    print("     for orders (0,0) / (-1,0) / (-4,0) / (-4,-2) -- monotone in the")
    print("     chief-ray offset the order reaches the last groups with.")
    print("     WHERE THAT LOSS IS *NOT* (niche D7, 2026-07-29, all measured):")
    print("     not apply_real_lens_traced's off-centre ray fit (aliasing-free")
    print("     exit-slope error 0.90 urad at 0.97 beam radii against 1.28 urad")
    print("     on axis untilted / 0.64 tilted -- so NOT uniformly below the")
    print("     on-axis figure; = 0.007 um of blur on a 3.5 um FWHM), not the fine")
    print("     retrace grid (NFC 12288 vs 16384: EE3 65.26 vs 65.26), not the")
    print("     Newton cap (12 vs 40 iters: 65.26), not the readout window (WF")
    print("     4/6/8: 65.26), not the coarse grid (RN 1024/2048/4096: 0.15 pt).")
    print("     By elimination it is the chain's TILTED-CONGRUENCE TRANSPORT")
    print("     across the coarse legs.  D7 raised the off-centre ray-fit order")
    print("     to 10 and that alone bought EE3 +4.3 / +4.1 / +4.8 points on the")
    print("     three off-axis rows (on axis: byte-identical).  An earlier")
    print("     revision of this banner blamed 'spurious coma, 3.7 -> 408 urad'")
    print("     -- that curve was decentred_fit_defect.py's own FFT-derivative")
    print("     measurement artefact; see that script's header.")
    print("     The POWER columns below (design%/FIELD%/ratio/throughput/")
    print("     capture) are the acceptance metric and are NOT affected.")

    print()
    print("frame table: (m,n)  design%  FIELD%  ratio  exact x,y (um)"
          "  throughput  capture  FWHM(um)  EE3%  EE6%  EE12%")
    for c, pd_, pm_, xy, fw, e3, e6, e12 in zip(
            res.congruences, share_des, cell_share, exact_c, fwhms, ee3s, ee6s,
            ee12s):
        print(f"  {c['name']:>9s}  {pd_ * 100:6.3f}  {pm_ * 100:6.3f}  "
              f"{pm_ / pd_:6.4f}  ({xy[0] * 1e6:+9.1f},{xy[1] * 1e6:+9.1f})  "
              f"{c['throughput']:.5f}  {c['capture']:.5f}  {fw * 1e6:7.3f}  "
              f"{e3 * 100:5.1f}  {e6 * 100:5.1f}  {e12 * 100:5.1f}")
    raise SystemExit(0 if all(v for _n, v in _checks) else 1)

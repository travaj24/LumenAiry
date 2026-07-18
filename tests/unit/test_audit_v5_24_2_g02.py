"""Regression tests for AUDIT_V5_24_2 exhaustive-audit group G02
(real bugs + dead code).

Covered findings:

* S3-9  -- the differential (analytic) ray trace's transfer-leg OPL must be
  SIGNED (drop ``abs``), matching the production ray tracer's RT-1 convention
  for overlapping-sag / negative-gap legs (``raytrace/differential.py``).
* S4-13 -- the interferometry phase-step round-trip must actually APPLY each
  reference shift and UNPACK the ``(phase, modulation)`` tuple; the GUI
  "Extract" button was silently meaningless (``analysis/interferometry.py`` +
  ``ui/interferometry_dock.py``).
* S4-14 -- codegen must write UTF-8 so non-latin glass / system names do not
  raise a locale UnicodeEncodeError (``io/codegen.py``).
* S4-16 -- listed dead code removed; the live paths still import and work.

Each S3-9 / S4-13 / S4-14 test uses an INDEPENDENT oracle (the production ray
tracer, the known input phase, and a locale-forced write) so a re-introduction
of the defect fails the test.
"""
import locale

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# S3-9 -- signed transfer-leg OPL in the differential (analytic) ray trace
# ---------------------------------------------------------------------------
from lumenairy.raytrace import (
    ray_transfer_jacobian,
    ray_transfer_jacobian_analytic,
    surfaces_from_prescription,
)

LAM = 0.633e-6


def test_s3_9_overlapping_sag_opl_is_signed_matches_main_trace():
    """Overlapping-sag geometry: a strongly convex surface with a THIN gap so
    off-axis rays intersect at a sag ``zi`` beyond the next vertex plane, giving
    a NEGATIVE transfer distance ``tau2 = (t - zi) / Np < 0``.  The analytic
    differential trace's base-ray OPL must then SUBTRACT the over-counted leg
    (signed ``n*t``), matching the production ray tracer.

    Independent oracle: the finite-difference ``ray_transfer_jacobian`` whose
    OPL is accumulated by ``trace()`` (the signed RT-1 ``_transfer``), a
    completely separate implementation from ``_adrt_step``.  Pre-fix the
    analytic path used ``abs(tau2)`` and diverged by millimetres.
    """
    presc = {'name': 'ov', 'aperture_diameter': 24e-3, 'surfaces': [
        {'radius': 10e-3, 'conic': 0., 'glass_before': 'air',
         'glass_after': 'N-BK7', 'semi_diameter': 12e-3},
        {'radius': -60e-3, 'conic': 0., 'glass_before': 'N-BK7',
         'glass_after': 'air', 'semi_diameter': 12e-3}],
        'thicknesses': [0.5e-3, 30e-3]}
    x = np.array([0.0, 3e-3, 5e-3, 6e-3, -6e-3])
    y = np.array([0.0, 0.0, 0.0, 2e-3, -2e-3])
    z = np.zeros(5)
    surfs = surfaces_from_prescription(presc)
    a = ray_transfer_jacobian_analytic(x, y, z, z, surfs, LAM)
    f = ray_transfer_jacobian(x, y, z, z, surfs, LAM)
    assert a.alive.all() and np.array_equal(a.alive, f.alive)
    # Primary discriminator: analytic OPL equals the production-trace OPL.
    assert np.max(np.abs(a.opd - f.opd)) < 1e-12
    # Non-vacuity: the marginal rays genuinely take the negative-tau2 branch --
    # their base-ray OPL is negative only because the transfer leg is SUBTRACTED
    # (with the pre-fix abs those OPLs were positive).
    assert a.opd[2] < -1e-5 and a.opd[3] < -1e-5


def test_s3_9_coordbreak_negative_gap_opl_is_signed():
    """A coordinate break with a NEGATIVE gap (reverse fold) drives the
    ``_adrt_coordbreak`` transfer distance ``tau < 0``.  Its OPL leg must be
    signed too (matching ``trace._apply_coord_break`` -> signed ``_transfer``);
    pre-fix ``abs(tau)`` over-counted the fold.  Oracle: the FD primitive."""
    from lumenairy.raytrace.surface import Surface
    base = surfaces_from_prescription({
        'name': 's', 'aperture_diameter': 20e-3, 'surfaces': [
            {'radius': 51.5e-3, 'conic': 0., 'glass_before': 'air',
             'glass_after': 'N-BK7', 'semi_diameter': 10e-3},
            {'radius': -51.5e-3, 'conic': 0., 'glass_before': 'N-BK7',
             'glass_after': 'air', 'semi_diameter': 10e-3}],
        'thicknesses': [4e-3, 10e-3]})
    cb = Surface(is_coordbrk=True, tilt_x_deg=1.0, tilt_y_deg=0.0,
                 tilt_z_deg=0.0, decenter_x_m=0.0, decenter_y_m=0.0,
                 thickness=-3e-3, glass_before='air', glass_after='air')
    det = Surface(radius=np.inf, glass_before='air', glass_after='air',
                  thickness=0.0)
    surfs = [base[0], base[1], cb, det]
    x = np.array([0.0, 3e-3, -4e-3])
    y = np.array([0.0, 2e-3, 1e-3])
    z = np.zeros(3)
    a = ray_transfer_jacobian_analytic(x, y, z, z, surfs, LAM)
    f = ray_transfer_jacobian(x, y, z, z, surfs, LAM)
    assert a.alive.all() and np.array_equal(a.alive, f.alive)
    assert np.max(np.abs(a.opd - f.opd)) < 1e-12


# ---------------------------------------------------------------------------
# S4-13 -- interferometry phase-step round-trip actually applies the shifts
# ---------------------------------------------------------------------------
from lumenairy.analysis.interferometry import (  # noqa: E402
    phase_shift_extract,
    phase_step_roundtrip,
    simulate_interferogram,
)

_PSI_LAM = 633e-9


def _opd_bowl(n=48):
    """A smooth defocus bowl scaled so the fringe phase stays within roughly
    (-1, 2) rad -- no 2*pi wrapping ambiguity, so the extracted phase equals the
    known input phase to machine precision."""
    t = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(t, t)
    return 2.0e-7 * (X ** 2 + Y ** 2 - 0.5)


@pytest.mark.parametrize("convention", ["hardware", "library"])
@pytest.mark.parametrize("steps", [3, 4, 5, 7])
def test_s4_13_phase_step_roundtrip_recovers_known_phase(convention, steps):
    """The PSI round-trip recovers the KNOWN input phase for either sign
    convention and any step count >= 3 (independent oracle: the analytic input
    phase)."""
    opd = _opd_bowl()
    phi, mod, shifts = phase_step_roundtrip(
        opd, _PSI_LAM, steps=steps, convention=convention)
    true_wrapped = np.angle(np.exp(1j * 2 * np.pi * opd / _PSI_LAM))
    diff = np.angle(np.exp(1j * (phi - true_wrapped)))
    rms = float(np.sqrt(np.mean(diff ** 2)))
    assert rms < 1e-9
    assert len(shifts) == steps
    assert np.all(mod > 0.0)


def test_s4_13_unapplied_shifts_are_meaningless():
    """Reproduce the pre-fix GUI defect: appending IDENTICAL frames (no shift
    ever applied) carries no phase diversity, so the extractor returns a flat
    (near-zero) phase and the residual is large.  The fixed round-trip -- which
    DOES apply the shifts -- recovers the phase.  Proves the shift application
    is load-bearing."""
    opd = _opd_bowl()
    steps = 4
    shifts = 2 * np.pi * np.arange(steps) / steps
    frame = simulate_interferogram(opd, _PSI_LAM, visibility=0.9)
    identical = np.asarray([frame] * steps)          # the bug: no shift applied
    phase, _mod = phase_shift_extract(
        identical, shifts=shifts, convention="library")
    true_wrapped = np.angle(np.exp(1j * 2 * np.pi * opd / _PSI_LAM))
    bad = np.angle(np.exp(1j * (phase - true_wrapped)))
    assert float(np.sqrt(np.mean(bad ** 2))) > 0.5   # meaningless recovery

    good_phi, _m, _s = phase_step_roundtrip(
        opd, _PSI_LAM, steps=steps, convention="library", visibility=0.9)
    ok = np.angle(np.exp(1j * (good_phi - true_wrapped)))
    assert float(np.sqrt(np.mean(ok ** 2))) < 1e-9


def test_s4_13_extract_returns_tuple_that_must_be_unpacked():
    """``phase_shift_extract`` returns a 2-tuple ``(phase, modulation)``.  The
    pre-fix GUI assigned it to a single name and subtracted, silently coercing
    the tuple to a ``(2, H, W)`` array.  Pin the tuple shape so the unpacking in
    ``phase_step_roundtrip`` is required."""
    opd = _opd_bowl(24)
    shifts = 2 * np.pi * np.arange(4) / 4
    frames = np.asarray([
        simulate_interferogram(opd + s * _PSI_LAM / (2 * np.pi), _PSI_LAM)
        for s in shifts])
    res = phase_shift_extract(frames, shifts=shifts, convention="library")
    assert isinstance(res, tuple) and len(res) == 2
    # The bug's failure mode: coercing the tuple to an array stacks phase and
    # modulation into a leading axis of size 2 (garbage when differenced).
    assert np.asarray(res).shape == (2,) + opd.shape


# ---------------------------------------------------------------------------
# S4-14 -- codegen writes UTF-8 (non-latin names must not crash the locale)
# ---------------------------------------------------------------------------
# Greek small lambda + two CJK ideographs -- all OUTSIDE cp1252 / ascii.  Kept
# as \u escapes so this source file stays ASCII (cp1252-safe).
_NONLATIN = "λ镜头"


def _codegen_prescription():
    from lumenairy import __version__  # noqa: F401 (ensures package import)
    return {
        "name": "Lens_" + _NONLATIN,
        "aperture_diameter": 25.4e-3,
        "wavelength": 1.31e-6,
        "elements": [],
        "surfaces": [],
        "thicknesses": [],
        "all_thicknesses": [],
        "all_glasses": [],
    }


def test_s4_14_codegen_writes_utf8_under_ascii_locale(tmp_path, monkeypatch):
    """Force the process's preferred encoding to ascii (so ``open(path, 'w')``
    with NO encoding would use ascii): writing a script whose system name has
    non-latin characters must still succeed, because codegen pins
    ``encoding='utf-8'``.  Pre-fix this raised UnicodeEncodeError.  The written
    file must round-trip the non-latin name as UTF-8."""
    from lumenairy.io.codegen import generate_simulation_script

    # Sanity: the chosen name is genuinely un-encodable in the forced locale, so
    # the test really does discriminate the fix.
    with pytest.raises(UnicodeEncodeError):
        _NONLATIN.encode("ascii")

    monkeypatch.setattr(locale, "getpreferredencoding", lambda *a, **k: "ascii")
    out = tmp_path / "generated.py"
    rx = _codegen_prescription()
    generate_simulation_script(
        rx, wavelength=1.31e-6, N=128, dx=25e-6, source_sigma=2e-3,
        output_path=str(out), include_plotting=False, include_analysis=False)
    # Bytes on disk must be valid UTF-8 carrying the non-latin name.
    text = out.read_text(encoding="utf-8")
    assert _NONLATIN in text


# ---------------------------------------------------------------------------
# S4-16 -- listed dead code removed; live paths still import + work
# ---------------------------------------------------------------------------
def test_s4_16_listed_dead_code_removed():
    """The eight symbols the audit flagged as dead are gone (guards against
    re-introduction)."""
    import lumenairy.elements.bor.zcascade as zc
    import lumenairy.elements.lenses_maslov as lm
    import lumenairy.elements.pmm.twod_jones as tj
    import lumenairy.elements.pmm.twod_staggered as ts
    import lumenairy.propagators.fga as fga
    import lumenairy.raytrace.surface as surf

    dead = [
        (fga, "_reconstruct"),
        (lm, "_gram_cho_factor"),
        (tj, "_assemble_2d_tensor"),
        (tj, "_require_inplane_tile"),
        (ts, "_axis_pair"),
        (ts, "_seg_outer_eps"),
        (zc, "cascade"),
        (surf, "_surface_sag_scalar"),
    ]
    for mod, name in dead:
        assert not hasattr(mod, name), (
            "%s.%s should have been removed as dead code" % (mod.__name__, name))
    assert "_surface_sag_scalar" not in surf.__all__


def test_s4_16_live_paths_survive_the_removals():
    """The removals must not have orphaned any live path: the sibling that only
    shares a name prefix stays, and the helpers ``cascade()`` used are still
    importable (``bor_solve`` re-uses them for its own inline cascade)."""
    # The helpers cascade() used stay live (bor_solve re-uses them inline);
    # pmm_jones_2d uses _require_nonzero_ezz, which stayed.
    import lumenairy.propagators.fga as fga
    from lumenairy.elements.bor.zcascade import (  # noqa: F401
        interface_smatrix,
        propagation_smatrix,
        redheffer_star,
    )
    from lumenairy.elements.pmm.twod_jones import pmm_jones_2d  # noqa: F401
    from lumenairy.raytrace.surface import Surface, _surface_sag_xy

    assert hasattr(fga, "_reconstruct_into")     # the live scatter wrapper
    # The surface-sag XY helper that replaced the scalar wrapper still works.
    s = Surface(radius=50e-3, conic=0.0, glass_before="air", glass_after="air")
    val = _surface_sag_xy(np.array([1e-3]), np.array([0.0]), s)
    assert np.all(np.isfinite(val))

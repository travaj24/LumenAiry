"""pytest plugin: run a test file with ONE engineered Phase D defect applied
for the whole session (the verifier's mutation matrix).  Select it with the
environment variable ``VD_MUT``:

  cd /c/tmp/lum_vcurved_d && VD_MUT=chi_T OMP_NUM_THREADS=1 ... \
    PYTHONPATH="C:/tmp/lum_vcurved_d;C:/tmp/lum_vcurved_d/validation/probe_pmm2d_curved/verify_d" \
    python -m pytest -p vd_mutplugin tests/unit/test_pmm2d_staggered_curved_d.py ...

Kinds: every ``_vdcommon.VMUT`` defect of the congruence kernel, plus
  raise_kernel  -- the three Phase D kernels raise (the unmapped suites must
                   stay green);
  drop_mu5      -- compile_shapes(with_mu=True) returns None as its mu;
  drop_paint    -- _paint_mu returns None (every route drops a shape's mu);
  d1_tol100     -- the TransfiniteMap derivative check's tolerance x100
                   (applied by wrapping numpy's max inside the check is not
                   possible from outside; this kind re-executes the class
                   body's check with the loosened bar through a source
                   patch of the module, see _loosen_d1)."""
import os

KIND = os.environ.get("VD_MUT", "")


def _loosen_d1():
    import inspect
    import textwrap

    import lumenairy.elements.pmm._curvemap as CM
    src = inspect.getsource(CM.TransfiniteMap.__init__)
    assert "1e-6 * max(scale" in src
    src = textwrap.dedent(src.replace("1e-6 * max(scale", "1e-4 * max(scale"))
    ns = {}
    exec(compile(src, CM.__file__, "exec"), CM.__dict__, ns)
    CM.TransfiniteMap.__init__ = ns["__init__"]


def pytest_configure(config):
    if not KIND or KIND == "none":
        return
    from lumenairy.elements.pmm import twod_staggered as TS
    if KIND == "raise_kernel":
        def boom(*a, **k):
            raise AssertionError("Phase D kernel reached")
        for nm in ("_stag_map_eff_tensor", "_stag_map_weights_tensor",
                   "_stag_map_as33"):
            setattr(TS, nm, boom)
        return
    if KIND == "drop_mu5":
        import lumenairy.elements.pmm as PM
        import lumenairy.elements.pmm.shapes2d as S2
        orig = S2.compile_shapes

        def cs(*a, with_mu=False, **k):
            out = orig(*a, with_mu=with_mu, **k)
            return out[:4] + (None,) if with_mu else out
        S2.compile_shapes = cs
        PM.compile_shapes = cs
        return
    if KIND == "drop_paint":
        import lumenairy.elements.pmm.shapes2d as S2
        S2._paint_mu = lambda *a, **k: None
        return
    if KIND == "d1_tol100":
        _loosen_d1()
        return
    from _vdcommon import vmutate
    cm = vmutate(KIND)
    cm.__enter__()
    config._vd_cm = cm


def pytest_unconfigure(config):
    cm = getattr(config, "_vd_cm", None)
    if cm is not None:
        cm.__exit__(None, None, None)


def pytest_report_header(config):
    return f"VD_MUT={KIND or '(none)'}"

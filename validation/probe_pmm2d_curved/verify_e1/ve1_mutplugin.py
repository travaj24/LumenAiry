"""pytest plugin: run test files with ONE engineered Phase E1 defect applied
for the whole session (the verifier's mutation matrix).  Select it with
``VE1_MUT``:

  cd /c/tmp/lum_vcurved_e1 && VE1_MUT=tau_J OMP_NUM_THREADS=1 ... \
    PYTHONPATH="C:/tmp/lum_vcurved_e1;C:/tmp/lum_vcurved_e1/validation/probe_pmm2d_curved/verify_e1" \
    python -m pytest -p ve1_mutplugin tests/unit/test_pmm2d_staggered_curved_e1.py ...

Kinds: every ``_ve1mut.KINDS`` defect, plus
  chitrap  -- ``_chi_R / _chi_Ktz / _chi_Gw`` RAISE when called on a MAPPED
              solver (the unmapped suites must stay green).
"""
import os

KIND = os.environ.get("VE1_MUT", "")


def pytest_configure(config):
    if not KIND or KIND == "none":
        return
    from lumenairy.elements.pmm import twod_staggered as TS
    if KIND == "chitrap":
        for nm in ("_chi_R", "_chi_Ktz", "_chi_Gw"):
            orig = getattr(TS.Granet2DTransverseE, nm)

            def wrap(self, *a, _o=orig, **k):
                if self.cmap is not None:
                    raise AssertionError("_chi_* reached under a map")
                return _o(self, *a, **k)
            setattr(TS.Granet2DTransverseE, nm, wrap)
        return
    from _ve1mut import vmutate
    cm = vmutate(KIND)
    cm.__enter__()
    config._ve1_cm = cm


def pytest_unconfigure(config):
    cm = getattr(config, "_ve1_cm", None)
    if cm is not None:
        cm.__exit__(None, None, None)


def pytest_report_header(config):
    return f"VE1_MUT={KIND or '(none)'}"

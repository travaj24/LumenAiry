"""TASK F -- a pytest plugin that REMOVES a gate's premise without editing the
library, so we can observe whether the gate FAILS, SKIPS or still passes.

Select the perturbation with the env var ``VF_PREMISE``:

  nodal_healthy   raise ``bor_solve._BOR_NODAL_SUPERUNITY_BAR`` above every
                  measured violation -- i.e. an arm on which the legacy nodal
                  cascade does NOT blow up past the bar.
  nodal_mild      raise the bar to 3e-2, just above the mildest broken row --
                  the smallest perturbation that removes the pathology from
                  ONE row (the test_bor_solve floor stack).
  sem_cold        raise ``_sem_contract._BOR_Q_EXCESS`` to 1e12 -- an arm on
                  which the manufactured sliver does NOT inject a hot
                  spectrum, so the refusal's second conjunct never fires.
  sem_hot         lower ``_BOR_Q_EXCESS`` to 1e2 -- an arm on which ORDINARY
                  geometry already reads "hot".
  eme_band        shrink ``_branch._EME_CUT_BAND_REL`` to 1e-18 -- an arm on
                  which the on-cut band no longer covers the backward error
                  (the pre-fix pathology, restored through the constant).
  none            no perturbation (control).
"""
from __future__ import annotations

import os

MODE = os.environ.get("VF_PREMISE", "none")


def pytest_configure(config):
    import lumenairy
    print("\n[vf_plugin_premise] lumenairy.__file__ =", lumenairy.__file__)
    print("[vf_plugin_premise] VF_PREMISE =", MODE)
    if MODE == "none":
        return
    if MODE in ("nodal_healthy", "nodal_mild"):
        from lumenairy.elements.bor import bor_solve as _bs
        new = 1.0e4 if MODE == "nodal_healthy" else 3.0e-2
        print("[vf_plugin_premise] _BOR_NODAL_SUPERUNITY_BAR %g -> %g"
              % (_bs._BOR_NODAL_SUPERUNITY_BAR, new))
        _bs._BOR_NODAL_SUPERUNITY_BAR = new
        _bs._BOR_NODAL_SUPERUNITY_WARN = new / 1e3
    elif MODE in ("sem_cold", "sem_hot"):
        from lumenairy.elements.bor import _sem_contract as _sc
        new = 1.0e12 if MODE == "sem_cold" else 1.0e2
        print("[vf_plugin_premise] _BOR_Q_EXCESS %g -> %g"
              % (_sc._BOR_Q_EXCESS, new))
        _sc._BOR_Q_EXCESS = new
    elif MODE == "bor_band":
        from lumenairy.elements.bor import _orient as _or
        kw = dict(_or.forward_orient.__kwdefaults__ or {})
        assert "band" in kw, kw
        kw["band"] = 1e-18
        _or.forward_orient.__kwdefaults__ = kw
        _or._BOR_CUT_BAND_REL = 1e-18
        print("[vf_plugin_premise] forward_orient band -> 1e-18 (the band no "
              "longer reaches the backward error: the defect, restored)")
    elif MODE == "eme_band":
        from lumenairy.elements.eme import _branch as _eb
        print("[vf_plugin_premise] _EME_CUT_BAND_REL %g -> 1e-18"
              % (_eb._EME_CUT_BAND_REL,))
        _eb._EME_CUT_BAND_REL = 1e-18
        # ``band=`` is KEYWORD-ONLY on both functions, so its default lives in
        # __kwdefaults__, not __defaults__.  Rebinding the module attribute
        # alone changes nothing -- the default was captured at def time.
        for fn in (_eb.cut_band, _eb.forward_decaying_root):
            kw = dict(fn.__kwdefaults__ or {})
            assert "band" in kw, (fn, kw)
            kw["band"] = 1e-18
            fn.__kwdefaults__ = kw
            print("[vf_plugin_premise]   %s.__kwdefaults__ = %s"
                  % (fn.__name__, fn.__kwdefaults__))
    else:
        raise SystemExit("unknown VF_PREMISE %r" % (MODE,))

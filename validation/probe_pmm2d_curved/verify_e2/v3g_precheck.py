"""Merge refusal of each verifier pair / awkward geometry (no solve)."""
from _ve import dump
from v3g_fix import stack
from v3g_geoms import awkward, pairs

out = {}
allg = {k: v[:2] for k, v in pairs().items()}
allg.update(awkward())
for name, (s1, s2) in allg.items():
    try:
        st = stack([(0.25, s1, 1.0), (0.2, s2, 1.0)], 4)
        r = dict(fast=st._perlayer_fast_ok(), refusal=st._merge_refusal,
                 grids=[list(L["own"]["cell"].shape[:2]) for L in st._layers],
                 mapped=[L["own"]["cmap"] is not None for L in st._layers])
    except Exception as ex:  # noqa: BLE001 -- recorded
        r = dict(exception=f"{type(ex).__name__}: {ex}")
    out[name] = r
    print(name, r, flush=True)
dump("v3g_precheck", out)

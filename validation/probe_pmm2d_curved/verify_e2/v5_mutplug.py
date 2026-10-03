"""pytest plugin (-p v5_mutplug) applying ONE default mutation, chosen by
the environment variable V5_MUT, by in-process monkeypatching:
  qoff        the q-matching block removed (shape layers keep the stack M)
  qall        q-matching over-applied (also to layers that named n_modes)
  rideoff     riding removed (_perlayer_geometry returns every own grid)
  rideall     riding over-applied (a uniform layer that NAMED n_modes rides)
"""
import os

from lumenairy.elements.pmm import stack2d_pure as SP

MUT = os.environ.get("V5_MUT", "")
_mc = SP.PMM2DStackPure._perlayer_modal_counts
_geo = SP.PMM2DStackPure._perlayer_geometry


def _qoff(self):
    Ms = _mc(self)
    return [int(L["M"]) if (L.get("own") is not None
                            and not L["own"].get("homogeneous")
                            and not L.get("pl_keywords")) else m
            for L, m in zip(self._layers, Ms)]


def _qall(self):
    saved = [L.get("pl_keywords") for L in self._layers]
    try:
        for L in self._layers:
            if L.get("own") is not None:
                L["pl_keywords"] = False
        return _mc(self)
    finally:
        for L, s in zip(self._layers, saved):
            if s is None:
                L.pop("pl_keywords", None)
            else:
                L["pl_keywords"] = s


def _rideoff(self, Ms):
    old = getattr(self, "_e2_no_ride", False)
    self._e2_no_ride = True
    try:
        return _geo(self, Ms)
    finally:
        self._e2_no_ride = old


def _rideall(self, Ms):
    saved = [L.get("pl_keywords") for L in self._layers]
    try:
        for L in self._layers:
            if L["kind"] in ("uniform", "uniform_tensor"):
                L["pl_keywords"] = False
        return _geo(self, Ms)
    finally:
        for L, s in zip(self._layers, saved):
            if s is None:
                L.pop("pl_keywords", None)
            else:
                L["pl_keywords"] = s


if MUT == "qoff":
    SP.PMM2DStackPure._perlayer_modal_counts = _qoff
elif MUT == "qall":
    SP.PMM2DStackPure._perlayer_modal_counts = _qall
elif MUT == "rideoff":
    SP.PMM2DStackPure._perlayer_geometry = _rideoff
elif MUT == "rideall":
    SP.PMM2DStackPure._perlayer_geometry = _rideall
print("v5_mutplug:", MUT or "none")

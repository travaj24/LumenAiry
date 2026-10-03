"""V3 -- STRUCTURE (C1): rectangles-only shapes= take the UNMAPPED solver,
and the F-B4 incident fix is reachable ONLY with a map.

Mutant: every SOLVER-side function that only a map can need is replaced by a
trap that records its name and raises.  Shapes are compiled (add_layer)
BEFORE the traps go in for the stack runs; the convenience entry (one call)
runs with the solver-side traps only.  Positive control: a curved shape must
fire the traps.

  python v3_struct.py   -> v3_struct_<build>.json
"""
import warnings

import numpy as np
from _vc import BUILD, dump

from lumenairy.elements.pmm import (
    Circle,
    PMM2DStackPure,
    Rect,
    _curvemap as CM,
    pmm_jones_2d_staggered,
    stack2d_pure as SP,
    twod_staggered as TS,
)
from lumenairy.elements.rcwa._core import uniaxial_tensor

warnings.simplefilter("ignore")
FIRED = []
SOLVER_FUNCS = [(TS, "_stag_map_nodes"), (TS, "_stag_map_weights"),
                (TS, "_stag_map_quad_rule"), (TS, "_stag_duffy_points"),
                (TS, "_stag_map_singular_corners"), (TS, "_stag_map_eff"),
                (TS, "_far_projector_mapped"),
                (TS, "_stag_incident_load_mapped"),
                (TS, "_stag_incident_coeffs_mapped"),
                (SP, "_stag_incident_coeffs_mapped")]
GEOM_FUNCS = [(CM.TransfiniteMap, "geom"), (CM.TransfiniteMap, "geom_points"),
              (CM.RefinedMap, "geom"), (CM.RefinedMap, "geom_points"),
              (CM.IdentityMap, "geom"), (CM.CellMap, "geom_points")]
SAVED = {}


def trap(name):
    def f(*a, **k):
        FIRED.append(name)
        raise RuntimeError(f"TRAP {name}")
    return f


def arm(geom=True):
    for mod, nm in SOLVER_FUNCS + (GEOM_FUNCS if geom else []):
        SAVED[(mod, nm)] = getattr(mod, nm)
        setattr(mod, nm, trap(f"{getattr(mod, '__name__', mod)}.{nm}"))


def disarm():
    for (mod, nm), f in SAVED.items():
        setattr(mod, nm, f)
    SAVED.clear()


P, WL = 1.2, 1.0
LC = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.4)
OUT = {}


def stack_rects(theta=0.0, phi=0.0, tensor=False, roundoff=False):
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=4, n_orders=2)
    st.add_layer(0.1, eps=2.1)
    if roundoff:
        # walls 0.2/0.7 x 0.45/0.75 and 0.85/1.05 x 0.15/0.45: the stripe's
        # straight edges are cut by other walls and their interior vertices
        # land 1 ulp off the grid (linear interpolation)
        sh = [Rect(0.45, 0.6, 0.5, 0.3, 4.0 + 0.2j),
              Rect(0.95, 0.3, 0.2, 0.3, LC if tensor else 2.25)]
    else:
        sh = [Rect(0.5, 0.6, 0.4, 0.2, 4.0 + 0.2j),
              Rect(1.0, 0.3, 0.2, 0.2, LC if tensor else 2.25)]
    st.add_layer(0.3, shapes=sh, background_eps=1.0)
    st.add_layer(0.2, shapes=[Rect(0.6, 0.6, 1.2, 0.4, 3.0)],
                 background_eps=1.5)
    st.set_source(WL, theta=theta, phi=phi)
    return st


for name, kw in {"rects_normal": {}, "rects_conical": dict(theta=0.3,
                                                          phi=0.5),
                 "rects_tensor_oblique": dict(theta=0.2, tensor=True),
                 "rects_roundoff_normal": dict(roundoff=True),
                 "rects_roundoff_tensor": dict(roundoff=True, tensor=True),
                 }.items():
    FIRED.clear()
    try:
        st = stack_rects(**kw)
    except Exception as e:                         # noqa: BLE001
        OUT[name] = dict(fired=[], solved=False, add_layer_raised=str(e)[:300])
        print(name, OUT[name])
        continue
    arm(geom=True)
    try:
        o, R, T, J = st.solve(retain_internal=True)
        ab = st.layer_absorption()
        OUT[name] = dict(fired=list(FIRED), solved=True,
                         unmapped=st.cmap is None,
                         closure=float(np.max(np.abs(
                             1 - R.sum(1) - T.sum(1) - np.sum(ab, 0)))))
    except Exception as e:                         # noqa: BLE001
        OUT[name] = dict(fired=list(FIRED), solved=False, err=str(e)[:200])
    disarm()
    print(name, OUT[name])

# the convenience entry: solver traps only (the merge itself evaluates the
# identity map's geometry to validate it)
FIRED.clear()
arm(geom=False)
try:
    out = pmm_jones_2d_staggered(P, P, None, 1.45, 1.0, 0.4, WL, n_modes=4,
                                 n_orders=2, theta=0.25, phi=0.1,
                                 shapes=[Rect(0.6, 0.6, 0.4, 0.4, 4.0)],
                                 background_eps=1.0)
    OUT["jones_rect"] = dict(fired=list(FIRED), solved=True)
except Exception as e:                             # noqa: BLE001
    OUT["jones_rect"] = dict(fired=list(FIRED), solved=False, err=str(e)[:200])
disarm()
print("jones_rect", OUT["jones_rect"])

# bytes: rectangles on uniform walls == the integer-grid eps_cell route
eps3 = np.ones((3, 3), complex)
eps3[1, 1] = 4.0
cmp = {}
for th, ph in ((0.0, 0.0), (0.3, 0.6)):
    a = pmm_jones_2d_staggered(P, P, None, 1.45, 1.0, 0.4, WL, n_modes=5,
                               n_orders=2, theta=th, phi=ph,
                               shapes=[Rect(0.6, 0.6, 0.4, 0.4, 4.0)],
                               background_eps=1.0)
    b = pmm_jones_2d_staggered(P, P, eps3, 1.45, 1.0, 0.4, WL, n_modes=5,
                               n_orders=2, theta=th, phi=ph)
    cmp[f"{th},{ph}"] = dict(
        bytes_equal=all(np.array_equal(np.asarray(x), np.asarray(y))
                        for x, y in zip(a[:3], b[:3])),
        maxdiff=float(max(np.max(np.abs(np.asarray(x) - np.asarray(y)))
                          for x, y in zip(a[1:3], b[1:3]))))
OUT["rect_vs_integer_grid"] = cmp
print("rect vs integer grid", cmp)

# positive control: a curved shape fires the solver traps
FIRED.clear()
st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=3,
                    n_orders=2)
st.add_layer(0.3, shapes=[Circle(0.6, 0.6, 0.36, 4.0)], background_eps=1.0)
st.set_source(WL)
arm(geom=False)
try:
    st.solve()
    OUT["control_circle"] = dict(fired=list(FIRED), solved=True)
except Exception as e:                             # noqa: BLE001
    OUT["control_circle"] = dict(fired=list(FIRED), solved=False,
                                 err=str(e)[:120])
disarm()
print("control", OUT["control_circle"])
# the incident fix alone: trap ONLY the incident helpers on a curved solve
FIRED.clear()
SAVED[(SP, "_stag_incident_coeffs_mapped")] = SP._stag_incident_coeffs_mapped
SP._stag_incident_coeffs_mapped = trap("SP._stag_incident_coeffs_mapped")
try:
    st.solve()
    OUT["control_incident_only"] = dict(fired=list(FIRED), solved=True)
except Exception:                             # noqa: BLE001
    OUT["control_incident_only"] = dict(fired=list(FIRED), solved=False)
disarm()
print("control incident", OUT["control_incident_only"])
dump(f"v3_struct_{BUILD}.json", OUT)

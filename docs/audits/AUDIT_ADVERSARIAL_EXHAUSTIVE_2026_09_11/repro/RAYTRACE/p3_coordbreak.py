"""RAYTRACE probe 6: coordinate-break conventions.

CONVENTIONS.md / world.py claim:
  world_R = Rx(+tx) @ Ry(+ty) @ Rz(+tz) is LOCAL->WORLD (Zemax KA-01638),
  so a tilt_x = +90 deg break puts the new local +z at world -y, and
  intersection._apply_coord_break applies its TRANSPOSE to rays.
"""
import sys
import numpy as np
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')
from lumenairy.raytrace.surface import Surface, RayBundle
from lumenairy.raytrace.intersection import _apply_coord_break
from lumenairy.raytrace.world import _apply_coord_break as world_cb, _rot_x, _rot_y, _rot_z

np.set_printoptions(precision=12, suppress=True)


def mk(x, y, z, L, M, N):
    a = np.atleast_1d
    n = 1
    return RayBundle(x=np.array([x]), y=np.array([y]), z=np.array([z]),
                     L=np.array([L]), M=np.array([M]), N=np.array([N]),
                     wavelength=1.0, alive=np.ones(1, bool), opd=np.zeros(1))


print('=' * 72)
print('1. world._apply_coord_break: tilt_x = +90 -> local +z in world coords')
print('=' * 72)
o, R = world_cb(np.zeros(3), np.eye(3), {'tilt_x_deg': 90.0})
print('  world_R[:,2] (local +z in world) =', R[:, 2],
      ' -> expected (0,-1,0)')
assert np.allclose(R[:, 2], [0, -1, 0], atol=1e-12), 'world site FAILED'
print('  PASS')

print()
print('=' * 72)
print('2. intersection._apply_coord_break: a +z-going ray after tilt_x=+90')
print('=' * 72)
r = mk(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
_apply_coord_break(r, Surface(is_coordbrk=True, tilt_x_deg=90.0))
d_local = np.array([r.L[0], r.M[0], r.N[0]])
print('  ray direction in NEW LOCAL frame =', d_local, ' -> expected (0,+1,0)')
print('  (docstring: "leaves a +z-going ray at local +y")')
# consistency: local dir must map back to the ORIGINAL world dir via world_R
d_world = R @ d_local
print('  world_R @ d_local =', d_world, ' -> must equal the input (0,0,1)')
ok = np.allclose(d_world, [0, 0, 1], atol=1e-14)
print('  CONSISTENT' if ok else '  *** INCONSISTENT ***')

print()
print('=' * 72)
print('3. Round-trip over 200 random (tx,ty,tz,dx,dy) and both PARM6 orders')
print('=' * 72)
rng = np.random.default_rng(3)
worst_d = 0.0
worst_p = 0.0
for trial in range(200):
    tx, ty, tz = rng.uniform(-60, 60, 3)
    dx, dy = rng.uniform(-5e-3, 5e-3, 2)
    order = int(rng.integers(0, 2))
    cb = dict(tilt_x_deg=tx, tilt_y_deg=ty, tilt_z_deg=tz,
              decenter_x_m=dx, decenter_y_m=dy, order=order)
    o_w, R_w = world_cb(np.zeros(3), np.eye(3), cb)
    # random ray in the OLD frame
    p0 = rng.uniform(-2e-2, 2e-2, 3)
    d0 = rng.normal(size=3); d0 /= np.linalg.norm(d0)
    r = mk(*p0, *d0)
    _apply_coord_break(r, Surface(is_coordbrk=True, tilt_x_deg=tx,
                                   tilt_y_deg=ty, tilt_z_deg=tz,
                                   decenter_x_m=dx, decenter_y_m=dy,
                                   coordbrk_order=order))
    p_loc = np.array([r.x[0], r.y[0], r.z[0]])
    d_loc = np.array([r.L[0], r.M[0], r.N[0]])
    # world.py says: r_old = origin + R @ r_local
    p_back = o_w + R_w @ p_loc
    d_back = R_w @ d_loc
    worst_p = max(worst_p, np.abs(p_back - p0).max())
    worst_d = max(worst_d, np.abs(d_back - d0).max())
print(f'  worst |position round-trip error| = {worst_p:.3e} m')
print(f'  worst |direction round-trip error| = {worst_d:.3e}')
print('  => the two sites ARE exact transposes/inverses'
      if max(worst_p, worst_d) < 1e-14 else
      '  *** the two sites DISAGREE ***')

print()
print('=' * 72)
print('4. PARM6 order semantics (decenter-then-tilt vs tilt-then-decenter)')
print('=' * 72)
for order in (0, 1):
    cb = dict(tilt_x_deg=30.0, decenter_y_m=1e-3, order=order)
    o_w, R_w = world_cb(np.zeros(3), np.eye(3), cb)
    print(f'  order={order}: world origin = {o_w}')
print('  Zemax: order 0 -> decenter applied in the OLD frame (0, 1mm, 0);')
print('         order 1 -> decenter applied in the NEW (tilted) frame.')

print()
print('=' * 72)
print('5. ui/model.py copy of the same transform (if present)')
print('=' * 72)
import subprocess
out = subprocess.run(
    ['grep', '-n', 'tilt', '-A3',
     r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy/lumenairy/ui/model.py'],
    capture_output=True, text=True)
print(out.stdout[:3000] if out.stdout else '(ui/model.py: no tilt hits / file absent)')

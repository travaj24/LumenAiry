import os, numpy as np, sys
os.environ.setdefault("JAX_ENABLE_X64","1")
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from lumenairy.elements.berreman import berreman_jones_1d

wl=633e-9
def rotz(a):
    c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def rotx(a):
    c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])

print("=== JAX/NumPy parity ===")
cases=[]
# in-plane uniaxial
Rz=rotz(np.pi/4); cases.append(("in-plane uniaxial", Rz@np.diag([1.5**2,1.6**2,1.5**2])@Rz.T))
# out-of-plane (tilted director)
Rx=rotx(0.5);      cases.append(("OOP tilted director", Rx@np.diag([1.5**2,1.6**2,1.5**2])@Rx.T))
# lossy OOP
Rx=rotx(0.5);      cases.append(("OOP lossy", Rx@np.diag([(1.5+0.05j)**2,(1.6+0.02j)**2,(1.5+0.05j)**2])@Rx.T))
for name,eps in cases:
    for thd in (0.0, 35.0):
        for phd in (0.0, 30.0):
            Rn,Tn,Jrn,Jtn = berreman_jones_1d([(eps,300e-9)], 1.5, 1.0, wl,
                    angle=np.radians(thd), phi=np.radians(phd))
            Rj,Tj,Jrj,Jtj = berreman_jones_1d([(jnp.asarray(eps,jnp.complex128),jnp.asarray(300e-9))],
                    jnp.asarray(1.5+0j), jnp.asarray(1.0+0j), jnp.asarray(wl),
                    angle=jnp.asarray(np.radians(thd)), phi=jnp.asarray(np.radians(phd)))
            dR=np.max(np.abs(np.asarray(Rj)-Rn)); dT=np.max(np.abs(np.asarray(Tj)-Tn))
            dJr=np.max(np.abs(np.asarray(Jrj)-Jrn)); dJt=np.max(np.abs(np.asarray(Jtj)-Jtn))
            print(f"  {name:22s} th={thd:4.1f} phi={phd:4.1f}: dR={dR:.2e} dT={dT:.2e} dJr={dJr:.2e} dJt={dJt:.2e}")

print("\n=== jax.grad finiteness (d R[0] / d thickness, / d eps) ===")
eps0 = rotx(0.5)@np.diag([1.5**2,1.6**2,1.5**2])@rotx(-0.5)
def f(t):
    R,T,Jr,Jt = berreman_jones_1d([(jnp.asarray(eps0,jnp.complex128), t)],
                                  jnp.asarray(1.5+0j), jnp.asarray(1.0+0j),
                                  jnp.asarray(wl), angle=jnp.asarray(0.4))
    return R[0]
g=jax.grad(f)(jnp.asarray(300e-9))
print("  dR/dt =", g, " finite:", np.isfinite(np.asarray(g)))
def f2(nr):
    R,T,Jr,Jt = berreman_jones_1d([(jnp.asarray(eps0,jnp.complex128), jnp.asarray(300e-9))],
                                  jnp.asarray(1.5+0j), jnp.asarray(1.0+0j),
                                  jnp.asarray(wl), angle=nr)
    return R[0]+T[0]
print("  d(R+T)/d(angle) =", jax.grad(f2)(jnp.asarray(0.4)), "(expect ~0 for lossless)")

print("\n=== jax x64 off behaviour ===")
print("  x64 enabled:", jax.config.jax_enable_x64)

print("\n=== NumPy native vs generalized path agreement (OOP at oblique) ===")
from lumenairy.elements import berreman as B
epsO = rotx(0.5)@np.diag([1.5**2,1.6**2,1.5**2])@rotx(-0.5)
Kx=1.0*np.sin(0.4); Ky=0.0
core=B._solve_core([epsO],[300e-9],1.0,1.5**2,wl,Kx,Ky)
Jr1,Jt1,R1,T1=B._farfield(core)
R2,T2,Jr2,Jt2=B._offplane_oblique_solve([epsO],[300e-9],1.0,1.5**2,wl,Kx,Ky)
print("  native R,T:",R1,T1)
print("  genrl  R,T:",R2,T2)
print("  max |dJt| =", np.max(np.abs(Jt1-Jt2)), " max|dJr| =", np.max(np.abs(Jr1-Jr2)))

"""Probe 1: exact OPD of a ray crossing a locally planar facet displaced by
`sag` along z -- the axial-translation identity (n2 pz2 - n1 pz1) * sag,
plus the walk W = s (p/pz1 - p_out/pz2).

Independent oracle: explicit 3-D ray trace through a tilted PLANE, OPL
referenced between two fixed planes z=0 (in) and z=Z (out).
"""
import numpy as np
import sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements._lens_real import _facet_axial_momenta

rng = np.random.default_rng(7)

def exact_plane_facet(n1, n2, p_t, gx, gy, s, Z, x0):
    """Exact 3-D trace: plane wave with transverse optical momentum p_t=(px,py)
    in medium n1 hits the plane z = s + gx*x + gy*y, refracts into n2, and
    we report the eikonal at the exit plane z=Z, for the ray that STARTS at
    transverse x0 on z=0.  Returns (S_exit, x_exit, p_out)."""
    px, py = p_t
    pz1 = np.sqrt(n1**2 - px**2 - py**2)
    d1 = np.array([px, py, pz1]) / n1              # unit direction
    nu = np.array([-gx, -gy, 1.0]); nu /= np.linalg.norm(nu)
    # ray r(t) = (x0[0], x0[1], 0) + t*d1; plane: nu . (r - P0) = 0, P0=(0,0,s)
    P0 = np.array([0.0, 0.0, s])
    r0 = np.array([x0[0], x0[1], 0.0])
    t_hit = np.dot(nu, P0 - r0) / np.dot(nu, d1)
    rh = r0 + t_hit * d1
    # exact vector Snell via momentum: p_out = p_in + Gamma * nu
    p_in = n1 * d1
    a = np.dot(p_in, nu)
    Gam = -a + np.sqrt(n2**2 - n1**2 + a*a)
    p_out = p_in + Gam * nu
    d2 = p_out / n2
    t2 = (Z - rh[2]) / d2[2]
    rexit = rh + t2 * d2
    S = n1 * t_hit + n2 * t2       # eikonal from z=0 plane point r0 to rexit
    return S, rexit, p_out

def run():
    worst_t1 = 0.0
    worst_walk = 0.0
    for trial in range(400):
        n1 = float(rng.uniform(1.0, 2.0))
        n2 = float(rng.uniform(1.0, 2.0))
        th = rng.uniform(0.0, 0.45)          # ray angle to z (rad)
        ph = rng.uniform(0, 2*np.pi)
        p_t = n1*np.sin(th)*np.array([np.cos(ph), np.sin(ph)])
        gx = float(rng.uniform(-0.30, 0.30))
        gy = float(rng.uniform(-0.30, 0.30))
        s  = float(rng.uniform(-2e-3, 2e-3))
        Z  = 5e-3
        # model: dz = pz2 - pz1 from _facet_axial_momenta
        dz, ok = _facet_axial_momenta(np.array(p_t[0]), np.array(p_t[1]),
                                      np.array(gx), np.array(gy), n1, n2, np)
        if not bool(ok):
            continue
        pz1 = np.sqrt(n1**2 - p_t[0]**2 - p_t[1]**2)
        pz2 = pz1 + float(dz)
        # ---- oracle: eikonal at the exit plane for the SAME transverse pixel
        # Screen model prediction:  S_model(x_exit) = p_t . x0 - OPD(x0) + ...
        # Cleanest test: compare the eikonal DIFFERENCE between facet at s and
        # facet at 0, both referenced to the same exit transverse point.
        x0 = np.array([0.0, 0.0])
        S_s, xe_s, po_s = exact_plane_facet(n1, n2, p_t, gx, gy, s, Z, x0)
        S_0, xe_0, po_0 = exact_plane_facet(n1, n2, p_t, gx, gy, 0.0, Z, x0)
        assert np.allclose(po_s, po_0)
        # both are plane waves with the same p_out; reference both to the SAME
        # exit point by removing p_out . (x_exit - x_ref)
        S_s_ref = S_s - np.dot(po_s[:2], xe_s[:2])
        S_0_ref = S_0 - np.dot(po_0[:2], xe_0[:2])
        dS = S_s_ref - S_0_ref            # == s*(pz1 - pz2) per eq (2)
        pred = s * (pz1 - pz2)
        err = abs(dS - pred) / max(abs(pred), 1e-30)
        worst_t1 = max(worst_t1, err)
        # ---- walk: the exit crossing of the vertex plane z=0 for the s-facet
        # ray that entered at x0 on z=0.  W = s*(p/pz1 - p_out/pz2) claimed.
        # exact: rh (hit) then travel back to z=0.
        d2 = po_s / n2
        t_back = (0.0 - (0.0 + (s + gx*0 + gy*0)*0))  # placeholder
        # recompute the hit point for the s facet:
        nu = np.array([-gx, -gy, 1.0]); nu /= np.linalg.norm(nu)
        d1 = np.array([p_t[0], p_t[1], pz1]) / n1
        t_hit = np.dot(nu, np.array([0,0,s]) - np.array([x0[0],x0[1],0.0]))/np.dot(nu, d1)
        rh = np.array([x0[0], x0[1], 0.0]) + t_hit*d1
        tb = (0.0 - rh[2]) / d2[2]
        rback = rh + tb*d2
        W_exact = rback[:2] - x0
        # model walk uses the facet height at the HIT point (s_hit) rather than s
        s_hit = rh[2]
        W_model = s_hit*(np.array(p_t)/pz1 - po_s[:2]/pz2)
        worst_walk = max(worst_walk, np.max(np.abs(W_exact - W_model))/max(np.max(np.abs(W_exact)),1e-30))
    print(f"axial-translation identity  (pz1-pz2)*s : worst rel err = {worst_t1:.3e}")
    print(f"walk  W = s_hit*(p/pz1 - p_out/pz2)     : worst rel err = {worst_walk:.3e}")

run()

import numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as lu
from lumenairy.elements import polarization as P

def jones_of(fn, **kw):
    """Extract the 2x2 Jones actually applied by an apply_* helper."""
    cols = []
    for e in ((1,0),(0,1)):
        jf = P.JonesField(np.array([[complex(e[0])]]), np.array([[complex(e[1])]]), 1e-6)
        out = fn(jf, **kw)
        cols.append([out.Ex[0,0], out.Ey[0,0]])
    return np.array(cols, dtype=complex).T   # columns = response to x,y

def A():
    # Stokes-from-coherency matrix:  S = A vec(J kron J*)  -- build A for the
    # library's Stokes convention S3 = -2 Im(Ex conj Ey).
    # vec order: (ExEx*, ExEy*, EyEx*, EyEy*)
    return np.array([[1,0,0,1],
                     [1,0,0,-1],
                     [0,1,1,0],
                     [0,1j,-1j,0]], dtype=complex)
# check: S2 = 2 Re(Ex Ey*) = (ExEy* + EyEx*) -> row (0,1,1,0) OK
#        S3 = -2 Im(Ex Ey*) = i(ExEy* - conj(ExEy*)) = i(ExEy*) - i(EyEx*)
#           -> row (0, i, -i, 0) OK

def mueller_from_jones(J):
    Amat = A()
    return np.real(Amat @ np.kron(J, J.conj()) @ np.linalg.inv(Amat))

def stokes_vec(Ex, Ey):
    return np.array([abs(Ex)**2+abs(Ey)**2, abs(Ex)**2-abs(Ey)**2,
                     2*np.real(Ex*np.conj(Ey)), -2*np.imag(Ex*np.conj(Ey))])

angles = np.linspace(0, np.pi, 8, endpoint=False)
print("=== 1. Jones matrices vs textbook ===")
maxdev = {}
for th in angles:
    c,s = np.cos(th), np.sin(th)
    R = np.array([[c,-s],[s,c]])
    # polarizer
    Jlib = jones_of(P.apply_polarizer, angle=th)
    Jtxt = R @ np.diag([1,0]).astype(complex) @ R.T
    maxdev['polarizer'] = max(maxdev.get('polarizer',0), np.abs(Jlib-Jtxt).max())
    # QWP (slow axis exp(+i delta), fast axis at th)
    for name, dlt in (('QWP', np.pi/2), ('HWP', np.pi), ('ret0.7', 0.7)):
        Jlib = jones_of(P.apply_waveplate, retardance=dlt, angle=th)
        Jtxt = R @ np.diag([1, np.exp(1j*dlt)]) @ R.T
        maxdev[name] = max(maxdev.get(name,0), np.abs(Jlib-Jtxt).max())
    # rotator
    Jlib = jones_of(P.apply_rotator, angle=th)
    maxdev['rotator'] = max(maxdev.get('rotator',0), np.abs(Jlib-R.astype(complex)).max())
for k,v in maxdev.items(): print(f"  {k:10s} max|J_lib - J_textbook| = {v:.3e}")

print("\n=== 2. Mueller M = A (J (x) J*) A^-1 vs textbook retarder Mueller ===")
# textbook (Collett) linear retarder, fast axis at th, retardance d, with the
# library S3 sign convention -> (S1,S2)<->S3 block NEGATED per module docstring
for th in (0.0, np.pi/8, np.pi/4, 0.3):
    d = np.pi/2
    J = jones_of(P.apply_waveplate, retardance=d, angle=th)
    M = mueller_from_jones(J)
    c2, s2 = np.cos(2*th), np.sin(2*th)
    cd, sd = np.cos(d), np.sin(d)
    Mtxt = np.array([
      [1,0,0,0],
      [0, c2*c2+s2*s2*cd, c2*s2*(1-cd), s2*sd],
      [0, c2*s2*(1-cd),   s2*s2+c2*c2*cd, -c2*sd],
      [0, -s2*sd, c2*sd, cd]])
    # library sign convention: negate the (S1,S2)<->S3 block
    Mconv = Mtxt.copy()
    Mconv[1:3,3] *= -1; Mconv[3,1:3] *= -1
    print(f"  th={th:.4f}: max|M_lib - M_textbook_convsigned| = {np.abs(M-Mconv).max():.3e}"
          f"   (raw textbook: {np.abs(M-Mtxt).max():.3e})")

print("\n=== 3. QWP fast axis +45deg on x-pol -> S3 ? (CONVENTIONS says -1) ===")
jf = P.JonesField(np.ones((2,2),complex), np.zeros((2,2),complex), 1e-6)
P.apply_quarter_wave_plate(jf, angle=np.pi/4)
S = P.stokes_parameters(jf)
print("  S3 =", S['S3'][0,0], " S0 =", S['S0'][0,0])
jf = P.JonesField(np.ones((2,2),complex), np.zeros((2,2),complex), 1e-6)
P.apply_quarter_wave_plate(jf, angle=-np.pi/4)
print("  fast axis -45deg -> S3 =", P.stokes_parameters(jf)['S3'][0,0])

print("\n=== 4. create_circular_polarized handedness ===")
for h in ('right','left'):
    jf = P.create_circular_polarized(np.ones((2,2),complex), 1e-6, h)
    print(f"  {h}: Ex={jf.Ex[0,0]:.4f} Ey={jf.Ey[0,0]:.4f} S3={P.stokes_parameters(jf)['S3'][0,0]:+.4f}")

print("\n=== 5. ellipse params round-trip ===")
for chi in (-0.7, -0.3, 0.0, 0.2, 0.7):
    for psi in (0.0, 0.4, 1.2):
        jf = P.create_elliptical_polarized(np.ones((1,1),complex), 1e-6, ellipticity=chi, orientation=psi)
        o,e = P.polarization_ellipse(jf)
        print(f"  chi={chi:+.2f} psi={psi:.2f} -> psi_out={o[0,0]:+.5f} chi_out={e[0,0]:+.5f}"
              f"  {'OK' if abs(e[0,0]-chi)<1e-12 and abs(((o[0,0]-psi+np.pi/2)%np.pi)-np.pi/2)<1e-12 else 'MISMATCH'}")

print("\n=== 6. rotation-matrix sign: Jones rotator vs Mueller rotation ===")
th=0.37
J = jones_of(P.apply_rotator, angle=th)
M = mueller_from_jones(J)
print("  Mueller of apply_rotator(th):\n", np.round(M,6))
print("  expected active Stokes rotation by 2th: S1'=c2 S1 - s2 S2 ... ")
print("  cos2th, sin2th =", np.cos(2*th), np.sin(2*th))

print("\n=== 7. PBS power conservation + port identity ===")
jf = P.create_linear_polarized(np.ones((1,1),complex), 1e-6, angle=0.3)
t,r = P.apply_polarizing_beam_splitter(jf, angle=0.0)
print("  |Et|^2+|Er|^2 =", (t.intensity()+r.intensity())[0,0], " input:", jf.intensity()[0,0])

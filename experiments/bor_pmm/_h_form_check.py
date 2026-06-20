"""Numerically check the H-formulation (H_r,H_phi) eigenproblem is the EXACT
dual of the E-formulation under eps<->1/eps, AND check whether spurious modes
move/persist. We use the validated E-solver as oracle and build the H-operator
by literal duality substitution.

Conventions (from the validated solver):
  exp(-iwt), mu=1, normalized h=sqrt(mu0/eps0)H, curl E = i k0 h, curl h=-i k0 eps E,
  gamma^2 = eps k0^2 - q^2.

Duality of source-free Maxwell with mu=1:
  curl E   = i k0 h          curl h = -i k0 eps E
  Swap (E -> h, h -> -E, eps -> 1/eps... let's verify) :
     curl h = i k0 (-E)?  We need curl h = -i k0 eps E. 
  The standard EM duality with mu=1: (E,H,eps,mu)->(H,-E,mu,eps). With mu=1 here
  and normalized h, the dual medium has eps' = 1 (mu plays role of eps). That is
  NOT eps<->1/eps. The H-formulation is NOT a trivial duality; eps stays in curl h.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from coupled_radial_eigensolver import radial_coupled_modes

m, Rbig, N, k0 = 1, 3.0, 200, 2.0
a = 1.0
def eps_profile(rr):
    # structured: two concentric rings (high contrast) to provoke spurious modes
    e = np.full_like(rr, 2.25, dtype=complex)
    e[rr < 0.6] = 12.0
    e[(rr >= 0.6) & (rr < 1.0)] = 2.25
    e[(rr >= 1.0) & (rr < 1.3)] = 12.0
    return e

modesE = radial_coupled_modes(m, Rbig, N, eps_profile, k0)
rd = np.array([md["reldiv"] for md in modesE])
qE = np.array([md["q"] for md in modesE])
print("E-FORM: total modes", len(modesE))
print("  spurious (reldiv>1):", int(np.sum(rd>1)), " physical (reldiv<0.5):", int(np.sum(rd<0.5)))
print("  real-q spurious (propagating-looking):",
      int(np.sum((rd>1) & (np.abs(qE.imag)<1e-3) & (qE.real>1e-2))))

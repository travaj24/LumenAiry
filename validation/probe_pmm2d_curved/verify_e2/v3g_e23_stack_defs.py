"""Definitions shared by v3g_e23_stack / v3g_e23_diag."""
import numpy as np
from v3g_fix import P

from lumenairy.elements.pmm import _curvemap as CM

x1, y1 = np.array([0, 0.2, 0.7, P]), np.array([0, 0.35, 0.95, P])
x2, y2 = np.array([0, 0.45, 1.0, P]), np.array([0, 0.1, 0.6, P])
E1, E2 = 3.0, 2.0
c1 = np.ones((3, 3), complex)
c1[1, 1] = E1
c2 = np.ones((3, 3), complex)
c2[1, 1] = E2
m1 = CM.SeparableStretch.from_physical_walls(
    x1, y1, fx=CM.SineStretch(0.05 * P), fy=CM.SineStretch(-0.07 * P))
m2 = CM.SeparableStretch.from_physical_walls(
    x2, y2, fx=CM.SineStretch(-0.04 * P), fy=CM.SineStretch(0.11 * P))



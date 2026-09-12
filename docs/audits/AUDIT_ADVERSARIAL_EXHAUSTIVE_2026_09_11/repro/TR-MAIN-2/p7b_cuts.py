import fixt, numpy as np
from lumenairy.elements import _lens_traced as LT
N, dx = 256, 6e-6
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x)
k = 2*np.pi/fixt.WL
g = np.exp(-(X**2+Y**2)/(0.25e-3)**2)
E = (g*np.exp(1j*k*25e-3*X) + g*np.exp(-1j*k*25e-3*X)).astype(np.complex128)
F = np.fft.fftshift(np.fft.fft2(E)); P = np.abs(F)**2
fx = np.fft.fftshift(np.fft.fftfreq(N, dx)); fy = fx
mx, my = P.sum(axis=0), P.sum(axis=1)
lox, hix = LT._occupied_freq_support(mx, fx, 0.995)
loy, hiy = LT._occupied_freq_support(my, fy, 0.995)
print('x band %.4g .. %.4g  (%d..%d)' % (lox, hix, np.searchsorted(fx,lox), np.searchsorted(fx,hix)))
print('y band %.4g .. %.4g' % (loy, hiy))
cutx = LT._spectral_gap_cuts(mx, fx, lox, hix, 0.15, 0.25)
cuty = LT._spectral_gap_cuts(my, fy, loy, hiy, 0.15, 0.25)
print('cutx', cutx)
print('cuty', cuty)
# characterise the y marginal near its cut
pn = my/my.max()
for c in cuty:
    i = int(np.argmin(np.abs(fy-c)))
    print('  y cut at index %d  f=%.4g  p/pk=%.3e   neighbours %.3e %.3e'
          % (i, c, pn[i], pn[i-1], pn[i+1]))
    print('  left max %.3e   right max %.3e' % (pn[:i].max(), pn[i+1:].max()))
print('y marginal: pk index', int(np.argmax(my)), ' p[0]/pk=%.2e p[-1]/pk=%.2e' % (pn[0], pn[-1]))
print('y marginal monotone-decreasing from peak?',
      bool(np.all(np.diff(pn[int(np.argmax(my)):]) <= 0)))
# how much power does the spurious y split move between bins?
segs = LT._segment_field_by_angle(E, dx, dx, 'auto', 'auto', 0.995, 0.15, 0.0, 32)
print('nseg', len(segs), 'powers', ['%.4f' % (float((np.abs(s)**2).sum())/float((np.abs(E)**2).sum())) for s in segs])

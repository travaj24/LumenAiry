import fixt, numpy as np, tracemalloc, time
from lumenairy.elements import _lens_traced as LT
for N in (512, 1024):
    dx = 6e-6
    x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x)
    k = 2*np.pi/fixt.WL
    g = np.exp(-(X**2+Y**2)/(0.25e-3)**2)
    E = (g*np.exp(1j*k*25e-3*X) + g*np.exp(-1j*k*25e-3*X)).astype(np.complex128)
    G = 8.0*N*N
    tracemalloc.start(); t=time.perf_counter()
    segs = LT._segment_field_by_angle(E, dx, dx, 'auto', 'auto', 0.995, 0.15, 0.0, 32)
    dt = time.perf_counter()-t
    _, pk = tracemalloc.get_traced_memory(); tracemalloc.stop()
    # 2-D vs 1-D window cost
    F = np.fft.fftshift(np.fft.fft2(E)); P = np.abs(F)**2
    fx = np.fft.fftshift(np.fft.fftfreq(N, dx))
    lox, hix = LT._occupied_freq_support(P.sum(axis=0), fx, 0.995)
    cutx = LT._spectral_gap_cuts(P.sum(axis=0), fx, lox, hix, 0.15, 0.25)
    FX, FY = np.meshgrid(fx, fx)
    W2 = LT._flattop_partition_1d(FX, cutx, 0.2*min(np.diff([lox]+sorted(cutx)+[hix])))
    W1 = LT._flattop_partition_1d(fx, cutx, 0.2*min(np.diff([lox]+sorted(cutx)+[hix])))
    print('N=%4d  nseg=%d  %.3fs  peak=%.1f MB = %.2f float64 grids' % (N, len(segs), dt, pk/1e6, pk/G))
    print('       window list: 2-D form %d x %.2f MB = %.2f MB ; 1-D form %d x %.6f MB'
          % (len(W2), W2[0].nbytes/1e6, sum(w.nbytes for w in W2)/1e6,
             len(W1), W1[0].nbytes/1e6))
    print('       separable check: max|W2[0]-W1[0][None,:] broadcast| = %.3e'
          % float(np.abs(W2[0] - np.broadcast_to(W1[0][None, :], (N, N))).max()))

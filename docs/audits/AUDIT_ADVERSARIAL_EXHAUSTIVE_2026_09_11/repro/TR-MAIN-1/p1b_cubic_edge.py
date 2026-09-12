import numpy as np
from scipy.ndimage import map_coordinates
N, sub, dx = 256, 8, 2e-6
x = (np.arange(N) - N/2)*dx
X = np.broadcast_to(x[None,:], (N,N)); Y = np.broadcast_to(x[:,None],(N,N))
Xs = X[::sub,::sub]; Ys = Y[::sub,::sub]
idx = np.arange(N, dtype=float)/sub
coords = np.empty((2,N,N)); coords[0]=idx[:,None]; coords[1]=idx[None,:]
for name, oc, ot in (("tilt", 1e-3*Xs, 1e-3*X),
                     ("defocus", (Xs**2+Ys**2)/0.2, (X**2+Y**2)/0.2)):
    for order in (1,3):
        up = map_coordinates(oc, coords, order=order, mode='nearest',
                             prefilter=(order>1))
        e = np.abs(up-ot)
        print(f"{name} order={order}")
        for m in (0,1,2,4,8):
            # error restricted to pixels at least m*sub from the +edge, i.e. core
            core = e[m*sub: (Xs.shape[0]-1-m)*sub+1, m*sub:(Xs.shape[0]-1-m)*sub+1]
            print(f"   inset {m} coarse cells: max={core.max()*1e9:9.4f} nm "
                  f"rms={np.sqrt((core**2).mean())*1e9:9.4f} nm")

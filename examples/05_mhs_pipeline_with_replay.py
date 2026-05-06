"""
Lumenairy example 5 -- MHS pipeline with checkpoint logging + replay.

Build an ASM -> prescription -> ASM MHS pipeline from a singlet, run
it with checkpoint logging to a tmp HDF5 store, then read every plane
back via replay_run() and verify shapes + labels.
"""
import os
import tempfile

import numpy as np
import lumenairy as la


def main():
    # --- 1. Build the pipeline from a prescription --------------------
    presc = la.make_singlet(
        R1=50e-3, R2=float('inf'), d=4e-3, glass='N-BK7',
        aperture=10e-3,
    )
    pipe = la.MhsPipeline.from_prescription(
        presc, wavelength=633e-9, dx=10e-6, Ny=64, Nx=64,
        pre_distance=5e-3, post_distance=5e-3, method='gbd',
        sample_step=4, chunk_beamlets=128,
    )
    print(f'  MHS pipeline: {pipe.n_subdomains} subdomains -> '
          f'{[s.label for s in pipe.subdomains]}')

    # --- 2. Source field ----------------------------------------------
    src = la.Source.gaussian(w0=200e-6, N=64, dx=10e-6, wavelength=633e-9)

    # --- 3. Run with checkpoint + store -------------------------------
    tmpdir = tempfile.mkdtemp()
    store_path = os.path.join(tmpdir, 'mhs_run.h5')
    seen = []

    def log_cb(idx, surface, field):
        seen.append((idx, surface.label, field.shape))

    final = pipe.run(
        src.E,
        return_intermediate=False,
        checkpoint=log_cb,
        store=store_path,
        label_prefix='example5',
    )
    print(f'  Pipeline complete; final field shape = {final.shape}')
    print(f'  Checkpoint saw {len(seen)} planes: '
          f'{[(idx, lbl) for idx, lbl, _ in seen]}')

    # --- 4. Replay from the store -------------------------------------
    # Note: passing wavelength here is purely metadata; the storage
    # format doesn't carry per-plane wavelength.
    result = la.replay_run(store_path, label_prefix='example5',
                            wavelength=633e-9, method='mhs-replay')
    print()
    print(f'  Replay result: {result}')
    print(f'  Replay history labels: {result.labels()}')

    # --- 5. Sanity check: replayed final equals live final ------------
    err = float(np.max(np.abs(result.field - final)))
    print(f'  Max |replay - live| at exit plane: {err:.3e}')

    # --- 6. Cleanup ---------------------------------------------------
    try:
        os.remove(store_path)
        os.rmdir(tmpdir)
    except OSError:
        pass


if __name__ == '__main__':
    main()

"""
Lumenairy example 3 -- high-fidelity wave-leg comparison.

Run the same prescription through three different wave propagators
(real_lens / GBD / HF) inside design_optimize and compare the
predicted Strehl.  Demonstrates the wave_propagator selector.
"""
import numpy as np
import lumenairy as la


def main():
    # Singlet with a moderate aspheric -- the case where propagation
    # method actually matters.
    template = {
        'name': 'aspheric singlet',
        'aperture_diameter': 8e-3,
        'surfaces': [
            {'radius': 25e-3, 'conic': -0.6,
             'aspheric_coeffs': {4: 5e5},
             'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'radius': float('inf'), 'conic': 0.0,
             'aspheric_coeffs': None,
             'glass_before': 'N-BK7', 'glass_after': 'air'},
        ],
        'thicknesses': [4e-3],
    }

    print(f'  Aspheric singlet, aperture=8 mm')
    print(f'  Comparing wave_propagator strategies on a Strehl evaluation:')
    print()

    # We're not optimizing here -- just doing one forward pass with
    # each propagator.  Use a no-op parameterization with a single
    # bound free variable that we don't move from its initial value.
    param = la.DesignParameterization(
        template=template,
        free_vars=[('surfaces', 0, 'radius')],
        bounds=[(20e-3, 30e-3)],
    )

    # Just need merit terms that read ctx.strehl_best so the wave leg
    # actually runs.  We don't need to converge -- one iteration is
    # enough for measurement.
    merits = [
        la.StrehlMerit(min_strehl=1.0, weight=1.0),
    ]

    print(f'  {"propagator":<14}  {"Strehl":>8}  {"merit":>10}  {"time(s)":>8}')
    print(f'  {"-"*14}  {"-"*8}  {"-"*10}  {"-"*8}')
    for wp in ('real_lens', 'gbd', 'hf'):
        try:
            res = la.design_optimize(
                parameterization=param, merit_terms=merits,
                wavelength=633e-9,
                N=64, dx=20e-6,
                method='L-BFGS-B', max_iter=1,
                wave_propagator=wp,
                verbose=False,
            )
            strehl = res.context_final.strehl_best
            print(f'  {wp:<14}  {strehl:>8.4f}  {res.merit:>10.3e}  '
                  f'{res.time_sec:>8.1f}')
        except Exception as e:
            print(f'  {wp:<14}  EXCEPTION: {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()

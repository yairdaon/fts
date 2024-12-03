from itertools import product

import numpy as np
import pandas as pd

BURN_IN_YEARS = 2000
COMMON = {
    'beta_slope': 0,
    'beta_change_start': 0,
    'burn_in_years': BURN_IN_YEARS,
    'run_years': 200
}

MEASLES = {
    'beta1': 1600,
    'beta2': 1600,
    #'dt_output': 7 / 365,
    'dt_euler': 1/ 10 / 365,
    'nu':  66,
    'psi': 1,
    'omega': 0,
    'mu': 1 / 50,
    'freq': 7,
    'what': 'measles'}

COBEY = {
    'beta1': 0.3,
    'beta2': 0.25,
    #'dt_output': 7,
    'dt_euler': 5e-2,
    'nu':  0.2,
    'psi': 360,
    'omega': 0,
    'mu': 1 / 30 / 360,
    'freq': 7,
    'what': 'cobey'}

def make_simulation_params(what,
                           pnas=None,
                           fast=False,
                           **kwargs):
    """Generate a list of dictionaries. Each dictionary is contains
    parameters for a two-strain simulation run. Any keyword argument
    you add will override parameters.

    """
    if what == 'measles':
        specific = MEASLES
        pnas_ = [1e-2] + [0, 1e-6, 1e-5, 1e-4, 1e-3] 
        eps1s = [0.18]
        eps2s = [0.18]
 
    elif what == 'cobey':
        specific = COBEY
        pnas_ = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1]  
        eps1s = [0.1]
        eps2s = [0.1]
    else:
        raise ValueError(f"No such thing as {what}!")
    
    pnas = pnas_ if pnas is None else pnas
    onas = [0]
    sigma21s = [0]
    sigma12s = [0.2, 0.8] if fast else np.round(np.arange(10) * 0.1, 3)
    params = []
    for eps1, eps2, pna, ona, sigma12, sigma21 in product(eps1s, eps2s, pnas, onas, sigma12s, sigma21s):
        p = {'pna': pna,
             'ona': ona,
             'sigma12': sigma12,
             'sigma21': sigma21,
             'eps1': eps1,
             'eps2': eps2,
             **specific}
        p = {**p, **COMMON} ## So COMMON overrides specific
        p = {**p, **kwargs} ## So kwargs overrides all other definitions
        params.append(p)
    return params[:3] if fast else params

    
def random_IC():
    """Generate random initial condition
    """    
    init = np.random.exponential(scale=1.0, size=6).reshape(3, 2)
    init = init / init.sum(axis=0)
    S_init = init[0, :]
    I_init = init[1, :]
    return S_init, I_init

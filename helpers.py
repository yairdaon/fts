from itertools import product

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression 

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
    'dt_output': 7 / 365,
    'dt_euler': 1/ 10 / 365,
    'nu':  66,
    'psi': 1,
    'omega': 0,
    'mu': 1 / 50}

COBEY = {
    'beta1': 0.3,
    'beta2': 0.25,
    'dt_output': 7,
    'dt_euler': 5e-2,
    'nu':  0.2,
    'psi': 360,
    'omega': 0,
    'mu': 1 / 30 / 360}

def make_simulation_params(what, pnas=None, fast=False, **kwargs):

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


def jacobian(X, how):
    x = X.fix
    y = X.shifted
    if how == 'OLS': ## Ordinary least squares
        x_norm = np.linalg.norm(x)
        return np.dot(x/x_norm, y/x_norm)
    elif how == 'TLS': ## Total least squares
        return np.corrcoef(x, y)[0,1]
    raise NotImplementedError(f"{how} is not implemented. Choose from OLS and TLS")
    
def rypdal_sugihara(x, window=12, shift=-1, how='OLS'):
    """
    The feature of Rypdal and Sugihara. 
    Returns numpy array L such that

    L[t] = argmin_l || l x[t:t-window] - x[t-shift:t-window-shift]  ||

    So that taking window =12 and shift = -1 gives a smoothed estimate
    for l such that x[t]l ~ x[t+1] at time t

    """
    ## x.shift(-1)[t] == x[t+1]: if x = [1,4,5,9] then x.shift(-1) = [4,5,9,nan]  
    dd = pd.DataFrame({'fix': x.values, 'shifted':x.shift(shift)}, index=x.index)
    
    L = np.full(dd.shape[0], np.nan)
    for i in range(window, dd.shape[0]):
        X = dd.iloc[i-window:i]
        L[i] = jacobian(X, how=how)
        # X = dd.iloc[i-window:i]
        # x = X.fix
        # y = X.shifted
        # lin = np.dot(x,y) / np.linalg.norm(x)**2
        # L[i] = lin 

        ## This assert statement succeeds
        # lm = LinearRegression(fit_intercept=False)
        # lin_ = lm.fit(x.values.reshape(-1,1), y).coef_[0]
        # assert np.abs(lin - lin_) < 1e-14
  

    # Dunno y this differs from L sometimes. It looks like it is doing the same
    #L_ = (dd.fix * dd.shifted).rolling(window).sum() / dd.fix.rolling(window).apply(np.linalg.norm)**2   

    return L#dd.assign(L=L, L_=L_.shift(-shift))

    
def random_IC():
    init = np.random.exponential(scale=1.0, size=6).reshape(3, 2)
    init = init / init.sum(axis=0)
    S_init = init[0, :]
    I_init = init[1, :]
    return S_init, I_init

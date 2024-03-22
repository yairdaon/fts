import numpy as np
import pandas as pd
import pdb

import pytest
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import time
from itertools import product

from fts.helpers import random_IC
from fts import original, multistrain
from fts.original import pilot

@pytest.mark.skip(reason="Mighty slow")
def test_pilot( ):
    """Run pilot tests to determine minimal step size"""    
    p = dict(beta0=[1600, 1333],
             mu=0.02,
             psi=1,
             eps=0.18,
             nu=66,
             dt_output=7,
             run_years=1e-12,     
             dt_euler=1e-2)

    steps = Parallel(n_jobs=-1, backend='loky', verbose=1)(
        delayed(pilot)(sigma12, **p) for _, sigma12 in product(range(5), np.arange(5) * 0.25))
    print("Minimal step size", min(steps))


def test_slow_fast():
    """Comparison of run time between our fast implementation and Cobey's 
    slower implementation
    """

    burn_in_years = 1
    run_years = 200
    dt_euler = 0.5
    S_init, I_init = random_IC()
    print(S_init)
    print(I_init)

    pna = 1e-2
    beta_slope = 0
    beta_change_start = 0
    dt_output = 30

    beta1 = 0.3
    beta2 = 0.25
    beta0 = np.array([beta1, beta2])
    sigma21 = 0
    sigma12 = 0.25

    n_pathogens = 2
    nu = 0.2
    psi = 360
    omega = 1
    eps = 0.1
    mu = 1 / 30 / psi
    np.random.seed(1324)
    test_noise = 100 * np.random.randn(2)
    
    ## Fast
    fast = multistrain.run(
        run_years=run_years,
        burn_in_years=burn_in_years,
        pna=pna,
        ona=0,
        dt_euler=dt_euler,
        n_pathogens=n_pathogens,
        dt_output=dt_output,
        S_init=S_init,
        I_init=I_init,
        mu=mu,
        nu=nu,
        beta1=beta1,
        beta2=beta2,
        sigma12=sigma12,
        sigma21=sigma21,
        beta_change_start=np.array([beta_change_start] * 2),
        beta_slope=np.array([beta_slope] * 2),
        psi=psi,
        omega=omega,
        eps=eps,
        test_noise=test_noise.copy())
    # fast = pd.DataFrame(index=np.arange(n_output+1) * dt_output/year,
    #                     data=logI,
    #                     columns=['I1', 'I2']) 

    # fast = fast.query("index > @burn_in_years")[['C1', 'C2']]
    # fast.plot()
    # plt.show()
    # pdb.set_trace()
        
    
    ## Slow
    start = time.time()
    slow = original.multistrain_sde(
        dt_euler=dt_euler,
        t_end=psi * (burn_in_years + run_years),
        n_pathogens=n_pathogens,
        dt_output=dt_output,
        S_init=S_init,
        I_init=I_init,
        mu=mu,
        nu=[nu] * 2,
        beta0=beta0,
        sigma=[[1, sigma12],
               [sigma21, 1]],
        beta_change_start=np.array([beta_change_start] * 2),
        beta_slope=np.array([beta_slope] * 2),
        psi=[psi] * 2,
        omega=[omega] * 2,
        eps=[eps] * 2,
        sd_proc=[pna] * 2,
        adaptive=False,
        gamma=[0, 0],
        corr_proc=0,
        test_noise=test_noise.copy())
    end = time.time()
    print("Native run time", (end - start) / 60, "minutes")
    t = np.array(slow['t'])
    slow = np.hstack([np.vstack(slow['logS']), np.vstack(slow['logI']), np.vstack(slow['CC']), np.vstack(slow['C'])])
    slow = pd.DataFrame(data=slow,
                        columns=['logS1', 'logS2', 'logI1', 'logI2', 'CC1', 'CC2', 'C1', 'C2'],
                        index=t)


    assert np.all(fast.index.values == slow.index.values)
    
    cols = ['logS1', 'logS2', 'logI1', 'logI2', 'CC1', 'CC2']
    diff1 = fast[cols] - slow[cols]
    err1 = np.max(np.abs(diff1).max())
    print("Maximal error", err1)
    assert err1 == 0

    
    diff2 = fast[['D1','D2']].values - slow[['C1', 'C2']].values
    err2 = np.max(np.abs(diff2))
    assert err2 == 0

    ## Test the forcing term is what we expect it to be
    # err = np.abs(fast.F1 / beta0[0] - 1) / eps - np.sin(2 * fast.index * np.pi).max()
    # assert err < 1e-14

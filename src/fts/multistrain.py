import pdb
import time

import numpy as np
import pandas as pd
from numba import jit, njit
from numba.core import types
from numpy import sin, cos, pi, log, exp, sqrt, ceil

from .helpers import random_IC, make_simulation_params

@jit(nopython=True)
def calc_log_betas(t,
                   beta0,
                   eps,
                   psi,
                   omega,
                   n_pathogens,
                   log_betas):
    for pathogen_id in range(n_pathogens):
        b = beta0[pathogen_id] * (1.0 + eps[pathogen_id] * sin(2.0 * pi / psi * (t - omega[pathogen_id] * psi)))
        log_betas[pathogen_id] = log(b)
    return log_betas

@jit(nopython=True)
def one_step(t,
             h,
             logS,
             dlogS,
             logI,
             dlogI,
             CC,
             dCC,
             log_mu,
             sigma,
             nu,
             beta0,
             eps,
             psi,
             omega,
             n_pathogens,
             log_betas,
             sd_proc,
             test_noise=None):

    mu = exp(log_mu)
    sqrt_h = sqrt(h)
    log_betas[:] = calc_log_betas(t, beta0, eps, psi, omega, n_pathogens, log_betas)
    
    noise = np.random.randn(n_pathogens) if test_noise is None else test_noise.copy()
    noise *= sd_proc

    dlogS[:] = 0
    dlogI[:] = 0
    dCC[:] = 0
            
    for i in range(n_pathogens):
        dlogS[i] += (exp(log_mu - logS[i]) - mu) * h

        for j in range(n_pathogens):
            if i != j:
                dlogSRij = sigma[i][j] * exp(log_betas[j] + logI[j])
                dlogS[i] -= dlogSRij * h
                dlogS[i] -= dlogSRij * noise[j] * sqrt_h
        dlogS[i] -= exp(log_betas[i] + logI[i]) * h
        dlogI[i] += exp(log_betas[i] + logS[i]) * h
        dCC[i] += exp(log_betas[i] + logS[i] + logI[i]) * h

        # Process noise
        dlogS[i] -= exp(log_betas[i] + logI[i]) * noise[i] * sqrt_h
        dlogI[i] += exp(log_betas[i] + logS[i]) * noise[i] * sqrt_h
        dCC[i] += exp(log_betas[i] + logS[i] + logI[i]) * noise[i] * sqrt_h
        dlogI[i] -= (nu[i] + mu) * h

    logS[:] = logS + dlogS
    logI[:] = logI + dlogI
    CC[:] = CC + dCC


@jit(nopython=True)
def multistrain_sde(
        dt_euler,
        t_end,
        dt_output, 
        n_pathogens, 
        logSs,
        logIs,
        CCs,
        Cs,
        Ts,
        Fs,
        mu,
        nu,
        beta0,
        beta_change_start,
        beta_slope,
        psi,
        omega,
        eps,
        sigma,
        sd_proc,
        dlogS,
        dlogI,
        dCC,
        log_betas,
        continuous_force,
        test_noise,
        seed):
    """ Numba code to run the two-strain model """
    if seed is not None:# and isinstance(seed, types.Integer):
        np.random.seed(seed)

    
    log_mu = log(mu)
               
    for output_iter in range(logSs.shape[0]-1):
        t = output_iter * dt_output
        t_next_output = t + dt_output

        log_betas[:] = calc_log_betas(t, beta0, eps, psi, omega, n_pathogens, log_betas)
        Fs[output_iter, :] = exp(log_betas)
        Ts[output_iter] = t

        ## Get "initial conditions" for this output cycle
        logS = logSs[output_iter, :]
        logI = logIs[output_iter, :]
        CC = CCs[output_iter, :]
       
        while t < t_next_output:
            t_next = min(t + dt_euler, t_next_output)

            one_step(t,
                     t_next - t,
                     logS,
                     dlogS,
                     logI,
                     dlogI,
                     CC,
                     dCC,
                     log_mu,
                     sigma,
                     nu,
                     beta0,
                     eps,
                     psi,
                     omega,
                     n_pathogens,
                     log_betas,
                     sd_proc,
                     test_noise=None)

                     
            t = t_next

           

        logSs[output_iter + 1, :] = logS
        logIs[output_iter + 1, :] = logI
        CCs[output_iter + 1, :] = CC
        Cs[output_iter + 1, :] = np.maximum(0, dCC)

    log_betas[:] = calc_log_betas(t, beta0, eps, psi, omega, n_pathogens, log_betas)
    Fs[-1, :] = exp(log_betas)
    Ts[-1] = t

def run(run_years,
        burn_in_years,
        beta1=0.3, 
        beta2=0.25,
        sigma12=0.25,
        sigma21=0,
        pna=0,
        ona=0, 
        beta_slope=0,
        beta_change_start=0,
        dt_output=30,
        dt_euler=5e-2,
        mu=1/30/360,
        nu=0.2,
        psi=360,
        omega=1,
        eps1=0.1,
        eps2=0.1,
        n_pathogens=2,
        test_noise=None,
        seed=None,#np.nan,
        S_init=None,
        I_init=None,
        continuous_force=True,
        drop_burn_in=False,
        **kwargs):
    """Native python code to allocate arrays for and appropriately pack
    the results of the numba code above
   
    Parameters:
    - run_years: int
        Simulation time user needs in years.
    - burn_in_years: int
        Initial period of simulation to discard (not considered in analysis). Total run time is run_years + burn_in_years
    - beta1: float
        Transmission force for the first pathogen.
    - beta2: float
        Transmission force for the second pathogen.
    - sigma12: float
        Cross-immunization that pathogen 2 elicits against pathogen 1.
    - sigma21: float
        Cross-immunization that pathogen 1 elicits against pathogen 2.
    - pna: float
        Process noise amplitude.
    - ona: float
        Observation noise amplitude.
    - beta_slope: float
        Slope of change in beta (transmission force) over time.
    - beta_change_start: int
        Year when change in beta starts.
    - dt_output: float
        Sampling time for output data.
    - dt_euler: float
        Time step for numerical integration using Eulers method.
    - mu: float, optional
        Natural mortality rate. Default is 1/(30*360), so if a year is 360 days then the average life span is 30 years.
    - nu: float, optional
        Recovery rate. Default is 0.2, so average time of illness and infectivity is 5 time units.
    - psi: float, optional
        Duration of a cycle of the environmental driver in "time
        units". So psi=360 means a full cycle (i.e. year) constitutes
        of 360 time units (i.e. days). OTOH psi=1 means the time unit
        is a year.
    - omega: float, optional
        Rate of change of betas. Ignore in current simulation, but you can dig in and play with it.
    - eps1: float
        Force of environmental driver effect on transmission for pathogen 1.
    - eps2: float
        Force of environmental driver effect on transmission for pathogen 2.
    - n_pathogens: int
        Number of pathogens in the simulation.
    - test_noise: np.array
        A noise array used for testing. User should ignore this.
    - seed: int, optional
        Seed for random number generator. Default is np.nan (no seed).
    - S_init: array
        Initial number of susceptibles.
    - I_init: array
        Initial number of infected individuals.
    - continuous_force: bool
        Whether to apply a continuous force (sine wave) of infection or a piecewise constant forcing.
    - drop_burn_in: bool
        Whether to exclude burn-in time from the returned time series.

    Returns:
    A pandas dataframe with the time series of susceptibles, infected, and possibly other states,
    depending on the implementation specifics.

    User should utilize C1 and C2 columns, which represent number of cases in the time before the sampling time. F1 and F2 the environmental drivers.
    """
    
    t_end = (run_years + burn_in_years) * psi
    n_output = int(ceil(t_end / dt_output))
    logS = np.empty((n_output + 1, 2))
    logI = np.empty((n_output + 1, 2))
    CC = np.zeros((n_output + 1, 2))
    C = np.zeros((n_output + 1, 2))
    F = np.ones((n_output + 1, 2))
    T = np.arange(n_output + 1) * dt_output

    if S_init is None or I_init is None:
        S_init, I_init = random_IC()
    logS[0, :] = log(S_init)
    logI[0, :] = log(I_init)
    beta0 = np.array([beta1, beta2], dtype=np.float64)
    sigma = np.array([[1, sigma12],
                      [sigma21, 1]])

    nu = np.full(n_pathogens, nu)
    omega = np.full(n_pathogens, omega)
    eps = np.array([eps1, eps2])
    beta_change_start = np.full(n_pathogens, beta_change_start)
    beta_slope = np.full(n_pathogens, beta_slope)

    start = time.time()
    multistrain_sde(
        dt_euler=dt_euler,
        t_end=t_end,
        dt_output=dt_output,
        n_pathogens=n_pathogens,
        logSs=logS,
        logIs=logI,
        CCs=CC,
        Cs=C,
        Ts=T,
        Fs=F,
        mu=mu,
        nu=nu,
        beta0=beta0,
        beta_change_start=beta_change_start,
        beta_slope=beta_slope,
        psi=psi,
        omega=omega,
        eps=eps,
        sigma=sigma,
        sd_proc=np.full(2, pna),
        dlogS=np.zeros(n_pathogens),
        dlogI=np.zeros(n_pathogens),
        dCC=np.zeros(n_pathogens),
        log_betas=np.empty(n_pathogens),
        continuous_force=continuous_force,
        test_noise=test_noise,
        seed=seed)
    end = time.time()
    # print("Simulation run time", (end - start) / 60, "minutes", flush=True)
    F[-1, :] = beta0 * (1 + eps * np.sin(2 * np.pi * T[-1]/psi))
    # logI += np.random.randn(*logI.shape) * ona
    cols = ['logS1', 'logS2', 'logI1', 'logI2', 'CC1', 'CC2', 'C1', 'C2', 'F1', 'F2']
    data = np.hstack([logS, logI, CC, C, F])
    df = pd.DataFrame(index=T, data=data, columns=cols)
    
    df['I1'] = np.exp(df.logI1) 
    df['I2'] = np.exp(df.logI2) 
    df['S1'] = np.exp(df.logS1) 

    df['S2'] = np.exp(df.logS2) 

    # This makes test pass with zero numerical error but it is not
    # good cuz it gives too many zeros and fucks up Rypdal-Sugihara
    df['D1'] = np.insert(np.diff(df.CC1.values),0, [0])
    df['D2'] = np.insert(np.diff(df.CC2.values),0, [0])    
    df['D1'] = np.maximum(0, df.D1)
    df['D2'] = np.maximum(0, df.D2)
    
    # df[['C1', 'C2']] = np.maximum(df[['C1', 'C2']].values, 0)

    df['C1C2'] = df.C1 + df.C2 / np.e
    df = df.query("index > @burn_in_years * @psi") if drop_burn_in else df
    return df



def measles(**kwargs):
    params = make_simulation_params(what='measles',
                                    **kwargs)[0]
    df = run(**params)
    df.index = pd.date_range(start='1900-01-01', periods=df.shape[0], freq='7d')
    df.index.name = 'time'
    return df
 

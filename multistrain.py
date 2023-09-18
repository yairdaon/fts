import pdb
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numba import jit
from numpy import sin, cos, pi, log, exp, sqrt, ceil

from fts.helpers import random_IC


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
    if not np.isnan(seed):
        np.random.seed(seed)

    pathogen_ids = range(n_pathogens)
    log_mu = log(mu)

    n_output = int(ceil(t_end / dt_output))

    logS = logSs[0, :]
    logI = logIs[0, :]
    CC = CCs[0, :]
    h = dt_euler

    for output_iter in range(n_output):
        t = output_iter * dt_output
        t_next_output = (output_iter + 1) * dt_output

        while t < t_next_output:
            t_next = t + h
            if t_next > t_next_output:
                t_next = t_next_output

            hh = t_next - t
            sqrt_hh = sqrt(hh)

            for pathogen_id in pathogen_ids:
                factor = (beta0[pathogen_id] + max(0.0, t - beta_change_start[pathogen_id]) * beta_slope[pathogen_id])
                if continuous_force:
                    cycle = sin(2.0 * pi / psi * (t - omega[pathogen_id] * psi))
                else:
                    cycle = 1 if t % 1 > 0.5 else 0
                b = factor * (1.0 + eps[pathogen_id] * cycle)
                log_betas[pathogen_id] = log(b)
            

            noise = np.random.randn(n_pathogens) if test_noise is None else test_noise.copy()
            noise *= sd_proc

            dlogS[:] = 0
            dlogI[:] = 0
            dCC[:] = 0
            
            for i in pathogen_ids:
                dlogS[i] += (exp(log_mu - logS[i]) - mu) * hh

                for j in pathogen_ids:
                    if i != j:
                        dlogSRij = sigma[i][j] * exp(log_betas[j] + logI[j])
                        dlogS[i] -= dlogSRij * hh
                        dlogS[i] -= dlogSRij * noise[j] * sqrt_hh
                dlogS[i] -= exp(log_betas[i] + logI[i]) * hh
                dlogI[i] += exp(log_betas[i] + logS[i]) * hh
                dCC[i] += exp(log_betas[i] + logS[i] + logI[i]) * hh

                # Process noise
                dlogS[i] -= exp(log_betas[i] + logI[i]) * noise[i] * sqrt_hh
                dlogI[i] += exp(log_betas[i] + logS[i]) * noise[i] * sqrt_hh
                dCC[i] += exp(log_betas[i] + logS[i] + logI[i]) * noise[i] * sqrt_hh
                dlogI[i] -= (nu[i] + mu) * hh

            logS = logS + dlogS
            logI = logI + dlogI
            CC = CC + dCC
            t = t_next

        logSs[output_iter + 1, :] = logS
        logIs[output_iter + 1, :] = logI
        CCs[output_iter + 1, :] = CC
        Cs[output_iter + 1, :] = np.maximum(0, dCC)
        Ts[output_iter + 1] = t
        Fs[output_iter, :] = exp(log_betas)


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
        seed=np.nan,
        S_init=None,
        I_init=None,
        continuous_force=True,
        drop_burnin=False,
        **kwargs):
    t_end = (run_years + burn_in_years) * psi
    n_output = int(ceil(t_end / dt_output))
    logS = np.empty((n_output + 1, 2))
    logI = np.empty((n_output + 1, 2))
    CC = np.zeros((n_output + 1, 2))
    C = np.zeros((n_output + 1, 2))
    F = np.ones((n_output + 1, 2))
    T = np.zeros(n_output + 1)

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
    df = df.query("index > @burn_in_years * @psi") if drop_burnin else df
    return df

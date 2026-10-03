import numpy as np
from scipy.stats import norm, gamma        
from channel import h_d, h_a, h_p, sample_noise   
import config                               

def generate_ar_process(tau, dt, n_samples, burn_in=1000, rng=None):
    """One AR(1) latent: correlated Gaussian, zero-mean unit-variance, coherence time tau."""
    if rng is None:
        rng = np.random.default_rng()

    phi = np.exp(-dt / tau)              # cling coefficient from coherence time
    sigma_kick = np.sqrt(1 - phi**2)     

    total = n_samples + burn_in
    z = np.empty(total)
    z[0] = rng.standard_normal()
    for t in range(1, total):
        z[t] = phi * z[t-1] + sigma_kick * rng.standard_normal()

    return z[burn_in:]                   # drop warm-up

def gaussian_to_gamma(z, shape):
    """Warp a unit-variance Gaussian sequence to a Gamma(shape) marginal (mean 1), rank-preserving."""
    u = norm.cdf(z)                              # gaussian value 
    g = gamma.ppf(u, a=shape, scale=1.0/shape)  # percentile 
    return g

def generate_ht(tau_s, tau_f, alpha, beta, n_samples, dt, rng=None):
    """Two-timescale Gamma-Gamma channel: warp a slow and a fast AR(1) process, multiply."""
    # alpha must be the bigger shape value (low shape = high variance, drowns the other timescale) - validated empirically
    slow_latent = generate_ar_process(tau_s, dt, n_samples, rng=rng)  # gaussian
    fast_latent = generate_ar_process(tau_f, dt, n_samples, rng=rng)  
    slow = gaussian_to_gamma(slow_latent, alpha)   # gamma
    fast = gaussian_to_gamma(fast_latent, beta)   
    ht = slow * fast                                # Gamma-Gamma, two timescales
    return ht

def generate_received_sequence(ht, system, L):
    """Assemble the received sequence r_e[t] = eta*(P*h_d*h_a*h_p[t]*h_t[t]) + n[t] along time."""
    # system params
    D_tx      = system["D_tx"]
    D_r       = system["D_r"]
    theta_div = system["theta_div"]
    tx_power  = system["tx_power"]

    n = len(ht)   

    hd = h_d(D_tx, D_r, theta_div, L)
    ha = h_a(config.CLEAR_ATTEN_DB_KM, L)

    hp = h_p(D_r, L, theta_div, config.THETA_JITTER, n)
    noise = sample_noise(n)

    r_e = config.ETA * (tx_power * hd * ha * hp * ht) + noise
    return r_e

def recover_estimate(r_e, system, L):
    """Divides out the deterministic factors to recover h_hat"""
    D_tx, D_r, theta_div, tx_power = system["D_tx"], system["D_r"], system["theta_div"], system["tx_power"]
    hd = h_d(D_tx, D_r, theta_div, L)
    ha = h_a(config.CLEAR_ATTEN_DB_KM, L)
    h_hat = r_e / (config.ETA * tx_power * hd * ha)
    return h_hat
import numpy as np
from scipy.stats import norm, gamma        
from channel import h_d, h_a, h_p, sample_noise   
import config                               
from scipy.stats import norm, gamma

def generate_ar_process(tau, dt, n_samples, burn_in=1000, rng=None):
    """Correlated Gaussian AR(1) latent, zero-mean unit-variance, coherence time tau."""
    if rng is None:
        rng = np.random.default_rng()

    phi = np.exp(-dt / tau)              # cling coefficient from coherence time
    sigma_kick = np.sqrt(1 - phi**2)     # keeps stationary variance = 1

    total = n_samples + burn_in
    z = np.empty(total)
    z[0] = rng.standard_normal()
    for t in range(1, total):
        z[t] = phi * z[t-1] + sigma_kick * rng.standard_normal()

    return z[burn_in:]                   # drop warm-up

def gaussian_to_gamma(z, shape):
    """Warp unit-variance Gaussian values to Gamma(shape, scale=1/shape) via inverse-CDF.
    Rank-preserving, so temporal correlation carries through. Mean of output = 1."""
    u = norm.cdf(z)                              # Gaussian value -> percentile [0,1]
    g = gamma.ppf(u, a=shape, scale=1.0/shape)  # percentile -> Gamma value (mean=1)
    return g
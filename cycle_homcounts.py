import numpy as np
from scipy.stats import poisson


def poisson_sample_max_size(lam, size=1, seed=42):
    """Sample from a Poisson distribution with mean lam, > 0."""
    rng = np.random.default_rng(seed)
    samples = poisson.rvs(lam, size=size*2, random_state=rng)
    samples = samples[samples > 0]
    while len(samples) < size:
        extra = poisson.rvs(lam, size=size, random_state=rng)
        samples = np.concatenate([samples, extra[extra > 0]])
    return samples[:size]

if __name__ == '__main__':
    sample = poisson_sample_max_size(20, size=1)[0]
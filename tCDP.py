import numpy as np

def compute_epsilon_from_rho_delta(rho_prime, omega, delta):
    """
    Convert tCDP parameters to (epsilon, delta) DP guarantee.
    """
    if omega <= 1:
        raise ValueError("omega must be > 1 for a valid tCDP to DP conversion.")
    Lambda = np.log(1 / delta)
    # Threshold for branch choice.
    threshold = ((omega - 1)**2)*rho_prime
    if Lambda <= threshold: # "small-delta" branch.
        epsilon = rho_prime + 2 * np.sqrt(rho_prime * Lambda)
    else: # "large-delta" branch.
        epsilon = rho_prime * omega + Lambda / (omega - 1)
    return epsilon

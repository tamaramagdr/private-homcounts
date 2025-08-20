import numpy as np


def sample_feature_mask(features, num_dimensions):
    """
    Sample a binary mask for feature perturbation.
    """
    n, d = features.shape
    selected_feature_indices = np.argpartition(np.random.rand(n, d), -num_dimensions, axis=1)[:, -num_dimensions:]

    # Create a boolean mask of shape (n, d).
    s = np.zeros((n, d), dtype=bool)
    row_indices = np.arange(n)[:, None]
    s[row_indices, selected_feature_indices] = True
    return s


def multi_bit_mechanism(features, epsilon, data_range=None, num_dimensions='max'):
    """
    Apply the multi-bit mechanism to perturb features.

    Adapted from:
    Sajadmanesh and Gatica-Perez, "Locally private graph neural networks."
    """
    # Get the number of features and their dimension.
    n, d = features.shape

    # If no data_range is provided, use the min and max of the features.
    if data_range is None:
        alpha = features.min(axis=0)
        beta = features.max(axis=0)
    else:
        alpha, beta = data_range

    # Determine the number of bits to use for perturbation.
    if num_dimensions == 'best':
        num_dimensions = int(max(1, min(d, np.floor(epsilon / 2.18))))
    elif num_dimensions == 'max':
        num_dimensions = d
    else:
        num_dimensions = int(num_dimensions)

    feature_mask = sample_feature_mask(features, num_dimensions)

    # Perturb sampled features.
    exp_mech = np.exp(epsilon / num_dimensions)
    scaled_features = (features - alpha) / (beta - alpha)
    # Replace NaN and None with 0 to avoid issues with log and exp.
    scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=1.0, neginf=0.0)
    # If not somethimes they seem to be out of range? Unclear.
    scaled_features = np.clip(scaled_features, 0, 1)
    perturb_prob = (scaled_features * (exp_mech - 1) + 1) / (exp_mech + 1)
    perturbed_bits = np.random.binomial(1, perturb_prob)

    # Unbiased results.
    signed_mask = feature_mask * (2 * perturbed_bits - 1)
    scaling_factor = d * (beta - alpha) / (2 * num_dimensions)
    reconstructed_features = scaling_factor * (exp_mech + 1) * signed_mask / (exp_mech - 1)
    reconstructed_features = reconstructed_features + (alpha + beta) / 2

    return reconstructed_features

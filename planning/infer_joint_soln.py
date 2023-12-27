import numpy as np
import scipy.linalg

# Yoinked from https://peterroelants.github.io/posts/gaussian-process-tutorial/
def exponentiated_quadratic(xa, xb):
    """Exponentiated quadratic  with σ=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, "sqeuclidean")
    return np.exp(sq_norm)


# Gaussian process posterior
def GP(X1, y1, X2, kernel_func):
    """
    Calculate the posterior mean and covariance matrix for y2
    based on the corresponding input X2, the observations (y1, X1),
    and the prior kernel function.
    """
    # Kernel of the observations
    Σ11 = kernel_func(X1, X1)
    # Kernel of observations vs to-predict
    Σ12 = kernel_func(X1, X2)
    # Solve
    solved = scipy.linalg.solve(Σ11, Σ12, assume_a="pos").T
    # Compute posterior mean
    μ2 = solved @ y1
    # Compute the posterior covariance
    Σ22 = kernel_func(X2, X2)
    Σ2 = Σ22 - (solved @ Σ12)
    return μ2, Σ2  # mean, covariance


# end yoink


def infer(
    samples_a: List[components.CompliantMotion],
    scores_a: List[float],
    samples_b: List[components.CompliantMotion],
) -> components.CompliantMotion:
    all_samples = samples_a + samples_b
    all_scores = np.array(scores_a + scores_b)
    mu_as = np.mean(all_scores)
    sigma_as_inv = 1.0 / np.std(all_scores)
    all_scores = sigma_as_inv * (all_scores - mu_as)
    # project all samples to 6d tangent space vectors

    # compute test_points
    test_points = np.array()

    # do regression
    test_points_out = GP(all_samples, all_scores, test_points, exponentiated_quadratic)

    # threshold and convert to compliant motion

    # validate

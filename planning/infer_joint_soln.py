import random
from typing import List

import numpy as np
import scipy.spatial
from pydrake.all import RigidTransform

import components
import mr

gen_infer = np.random.default_rng(0)

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
    all_samples: List[components.CompliantMotion],
    all_scores: List[float],
) -> List[components.CompliantMotion]:

    mu_as = np.mean(all_scores)
    sigma_as_inv = 1.0 / np.std(all_scores)
    all_scores = sigma_as_inv * (all_scores - mu_as)

    # project all samples to 6d tangent space vectors
    nominal = all_samples[0]
    all_samples_r6 = []
    for sample in all_samples:
        diff = nominal.X_WCd.InvertAndCompose(sample.X_WCd)
        t = diff.translation()
        r = mr.so3ToVec(mr.MatrixLog3(diff.rotation().matrix()))
        all_samples_r6.append(np.concatenate((r, t)))

    # compute test_points
    num_test_points = 100
    test_points = []
    for i in range(num_test_points):
        base = np.array(random.choice(all_samples_r6))
        noise = gen_infer.uniform(low=-0.02, high=0.02, size=6)
        test_points.append(base + noise)

    test_points = np.array(test_points)
    all_samples_r6 = np.array(all_samples_r6)

    # do regression
    K = 16
    test_points_out, _ = GP(
        all_samples_r6, all_scores, test_points, exponentiated_quadratic
    )
    ind = np.argpartition(test_points_out, -K)[-K:]
    top_n = test_points[ind]

    # threshold and convert to compliant motion
    U = []
    for pt_idx in range(top_n.shape[0]):
        pt = top_n[pt_idx]
        X = RigidTransform(mr.MatrixExp6(mr.VecTose3(pt)))
        U.append(
            components.CompliantMotion(
                nominal.X_GC, nominal.X_WCd.multiply(X), nominal.K
            )
        )

    return U

import random
import warnings
from typing import List

import numpy as np
import scipy.spatial
from pydrake.all import RigidTransform
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPRegressor

import components
import mr

warnings.simplefilter("ignore", category=ConvergenceWarning)
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
    Sigma11 = kernel_func(X1, X1)
    Sigma11_ridge = Sigma11 + 0.001 * np.eye(Sigma11.shape[0])
    # Kernel of observations vs to-predict
    Sigma12 = kernel_func(X1, X2)
    # Solve
    try:
        solved = scipy.linalg.solve(Sigma11_ridge, Sigma12, assume_a="pos").T
    except Exception as e:
        solved = scipy.linalg.solve(Sigma11_ridge, Sigma12).T
    # Compute posterior mean
    μ2 = solved @ y1
    # Compute the posterior covariance
    Σ22 = kernel_func(X2, X2)
    Σ2 = Σ22 - (solved @ Sigma12)
    return μ2, Σ2  # mean, covariance


# end yoink


def gp_sklearn(X1, y1, X2, kf):
    del kf
    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp.fit(X1, y1)
    mu, std = gp.predict(X2, return_std=True)
    return mu, std


def NN(X1, y1, X2):
    nn = MLPRegressor(hidden_layer_sizes=(15, 15))
    nn = nn.fit(X1, y1)
    predictions = nn.predict(X2)
    return predictions, None


def infer(
    all_samples: List[components.CompliantMotion], all_scores: List[float], do_gp: bool
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
    K = 16
    if do_gp:
        num_test_points = 1000
    else:
        num_test_points = K
    test_points = []
    for i in range(num_test_points):
        base = np.array(random.choice(all_samples_r6))
        noise = gen_infer.uniform(low=-0.02, high=0.02, size=6)
        test_points.append(base + noise)

    test_points = np.array(test_points)
    all_samples_r6 = np.array(all_samples_r6)

    # do regression
    test_points_out, _ = gp_sklearn(
        all_samples_r6, all_scores, test_points, exponentiated_quadratic
    )
    # test_points_out, _ = NN(all_samples_r6, all_scores, test_points)
    if do_gp:
        ind = np.argpartition(test_points_out, -K)[-K:]
        top_n = test_points[ind]
    else:
        top_n = test_points[:K]

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

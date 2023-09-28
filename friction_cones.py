from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydrake.all import RigidTransform


@dataclass(frozen=True)
class FrictionCone:
    mu: float
    Fn: np.ndarray


def normalize(v: np.ndarray) -> np.ndarray:
    assert np.linalg.norm(v) > 1e-10
    return v / np.linalg.norm(v)


def contact_to_cone(
    X_WP: RigidTransform, A: np.ndarray, b: np.ndarray, mu: float
) -> FrictionCone:
    """Given two rigid bodies in contact, come up with the corresponding friction cone.

    Given the pose of the manipuland in the world frame and the constraints that define
    the body in the environment that the manipuland is contacting, we first identify
    which plane on the object the manipuland is touching. Then we extract the corresponding
    normal vector and compute two orthogonal vectors which can be minkowski summed to get
    an underapproximation of the friction cone.

    """
    p = X_WP.translation()
    # the normal of the plane in contact can be found by looking for whichever
    # constraint is the "tightest" among the planes of the polyhedron.
    residuals = (A @ p) - b
    fn_idx = np.argmax(residuals)
    Fn_W = A[fn_idx]
    return FrictionCone(mu, normalize(Fn_W))


def project_wrench_to_cone(F: np.ndarray, fc: FrictionCone) -> np.ndarray:
    Fn_negative_scaled = (np.dot(F, -fc.Fn)) * -fc.Fn
    stiction_f = F - Fn_negative_scaled
    if np.linalg.norm(stiction_f) < 1e-6:
        print("warning: ambiguous sticking contact detected")
        return F
    stiction_f_normed = stiction_f / np.linalg.norm(stiction_f)
    min_sliding_force = fc.mu * np.linalg.norm(Fn_negative_scaled)
    desired_tangential_component = min_sliding_force * stiction_f_normed
    return F - stiction_f + desired_tangential_component


def simple_wrench_projection_example():
    ground_A = np.array([[0.0, 0.0, 1.0]])
    ground_b = np.array([0.0])
    fc = contact_to_cone(RigidTransform([0.0, 0.0, 0.0]), ground_A, ground_b, 1.0)
    F_curr = np.array([0.1, 0.0, -1.0])
    projection = project_wrench_to_cone(F_curr, fc)
    print(f"{projection=}")


def wrench_projection_example():
    ground_A = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, 0.0]])
    ground_b = np.array([0.0, 1.0])
    fc = contact_to_cone(RigidTransform([0.0, 0.0, 0.0]), ground_A, ground_b, 1.0)
    print(f"{fc.Fn=}")
    F_curr = np.array([0.1, 0.0, -1.0])
    projection = project_wrench_to_cone(F_curr, fc)
    print(f"{projection=}")


if __name__ == "__main__":
    wrench_projection_example()

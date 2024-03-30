from pydrake.all import RigidTransform

import components
import utils
from planning import infer_joint_soln


def test_gp():
    X_WGa1 = utils.xyz_rpy_deg([0.5, 0.0, 0.2], [180, 0, 0])
    X_WGa2 = utils.xyz_rpy_deg([0.505, 0.0, 0.2], [180, 0, 0])
    c_a_1 = components.CompliantMotion(RigidTransform(), X_WGa1, components.stiff)
    c_a_2 = components.CompliantMotion(RigidTransform(), X_WGa2, components.stiff)
    scores_a = [0, 0.5]

    X_WGb1 = utils.xyz_rpy_deg([0.53, 0.0, 0.2], [180, 0, 0])
    X_WGb2 = utils.xyz_rpy_deg([0.525, 0.0, 0.2], [180, 0, 0])
    c_b_1 = components.CompliantMotion(RigidTransform(), X_WGb1, components.stiff)
    c_b_2 = components.CompliantMotion(RigidTransform(), X_WGb2, components.stiff)
    scores_b = [0, 0.5]

    infer_joint_soln.infer([c_a_1, c_a_2], scores_a, [c_b_1, c_b_2], scores_b)


if __name__ == "__main__":
    test_gp()

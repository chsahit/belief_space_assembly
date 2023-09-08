import numpy as np
from pydrake.all import RigidTransform

import components
import dynamics
import state
import utils
import visualize
from simulation import ik_solver


def init(x_err=0.0, y_err=0.0, z=0.155, theta_err=0.0):
    X_WG_0 = utils.xyz_rpy_deg([0.50, 0.0, 0.36], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([x_err, 0.0, z], [0, theta_err, 0])
    X_WO = utils.xyz_rpy_deg([0.5, y_err, 0.075], [0, 0, 0])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p_0 = state.Particle(
        q_r_0, X_GM, X_WO, "assets/chamfered_hole.sdf", "assets/peg.urdf"
    )
    return p_0


def stiff_pih_no_rcc():
    p0 = init(y_err=-0.01)
    X_WG_d = utils.xyz_rpy_deg([0.50, 0.0, 0.2], [180, 0, 0])
    u0 = components.CompliantMotion(RigidTransform(), X_WG_d, components.stiff)
    p1 = dynamics.simulate(p0, u0, vis=True)


def stiff_pih_with_rcc():
    p0 = init(x_err=-0.01)
    X_WG_d = utils.xyz_rpy_deg([0.50, 0.0, 0.0], [180, 0, 0])
    u0 = components.CompliantMotion(RigidTransform([0.0, 0.0, 0.23]), X_WG_d, components.stiff)
    p1 = dynamics.simulate(p0, u0, vis=True)


def stiff_pih_theta_err():
    p0 = init(x_err=0.01, theta_err=-10)
    X_WG_d = utils.xyz_rpy_deg([0.50, 0.0, 0.0], [180, 0, 0])
    u0 = components.CompliantMotion(RigidTransform([0.0, 0.0, 0.23]), X_WG_d, components.mostly_soft)
    p1 = dynamics.simulate(p0, u0, vis=True)


if __name__ == "__main__":
    stiff_pih_no_rcc()

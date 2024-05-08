import numpy as np
from pydrake.all import RigidTransform

import components
import dynamics
import utils
from experiments import init_particle
from simulation import ik_solver


def rotate_hand():
    p0 = init_particle.init_peg()
    curr_X_WG = p0.X_WG
    delta = utils.xyz_rpy_deg([0, 0, 0], [20, 0, 0])
    X_WGd = curr_X_WG.multiply(delta)
    u1 = components.CompliantMotion(
        RigidTransform(), curr_X_WG, np.diag(components.stiff)
    )
    u2 = components.CompliantMotion(RigidTransform(), X_WGd, np.diag(components.stiff))
    ik_solver.update_motion_qd(u1)
    ik_solver.update_motion_qd(u2)
    utils.pickle_trajectory([u1, u2])
    dynamics.simulate(p0, u2, vis=True)
    input()


if __name__ == "__main__":
    rotate_hand()

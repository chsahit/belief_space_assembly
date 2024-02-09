import numpy as np
from pydrake.all import RigidTransform

import components
import dynamics
import utils
from experiments import init_particle
from simulation import ik_solver


def push():
    p = init_particle.init_peg(y=-0.04)
    X_WGd = utils.xyz_rpy_deg([0.50, 0.0, 0.22], [180, 0, 0])
    K = np.array([10.0, 10.0, 10.0, 100.0, 100.0, 400.0])
    u = components.CompliantMotion(RigidTransform(), X_WGd, components.stiff)
    ik_solver.update_motion_qd(u)

    dynamics.simulate(p, u, vis=True)


if __name__ == "__main__":
    push()
    input()

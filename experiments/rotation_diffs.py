import numpy as np
from pydrake.all import RigidTransform, Quaternion, RotationMatrix

import components
import dynamics
import state
import utils
import visualize
from simulation import ik_solver

def init(X_GM_x: float = 0.0):
    X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, 0.36], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([X_GM_x, 0.0, 0.155], [180, 0, 0])
    X_WO = utils.xyz_rpy_deg([0.5, 0, 0.075], [0, 0, 0])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p_0 = state.Particle(
        q_r_0, X_GM, X_WO, "assets/big_chamfered_hole.sdf", "assets/peg.urdf"
    )
    return p_0

def do_sims():
    p0 = init()
    v = np.array([-0.01, 1, 0, 0])
    rot1_d = Quaternion(v/np.linalg.norm(v))
    X_WG_d = RigidTransform(rot1_d, [0.55, 0.0, 0.36])
    u0 = components.CompliantMotion(RigidTransform(), X_WG_d, components.stiff)
    dynamics.simulate(p0, u0, vis=True)
    print("done")
    input()

if __name__ == "__main__":
    do_sims()

import numpy as np
from pydrake.all import RigidTransform

import components
import contact_defs
import dynamics
import state
import utils
from planning import refine_motion
from simulation import ik_solver


def p_nom() -> state.Particle:
    X_WG_0 = utils.xyz_rpy_deg([0.50, 0.0, 0.36], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([0.0, 0.0, 0.155], [0, 0, 0])
    X_WO = utils.xyz_rpy_deg([0.5, 0, 0.075], [0, 0, 0])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p_0 = state.Particle(
        q_r_0, X_GM, X_WO, "assets/chamfered_hole.sdf", "assets/peg.urdf"
    )
    return p_0


def b0_easy() -> state.Belief:
    p0 = p_nom()
    grasps = [
        components.Grasp(x=-0.01, z=0.155, pitch=0),
        components.Grasp(x=0.01, z=0.155, pitch=0),
    ]
    bin_poses = [components.ObjectPose(x=0.5, y=0, yaw=0)] * len(grasps)
    b = state.Belief.make_particles(grasps, bin_poses, p0)
    return b


def test_refine_motion():
    b0 = b0_easy()
    u = refine_motion.refine(b0, contact_defs.ground_align)
    b_result = dynamics.f_bel(b0, u)
    print(b_result.satisfies_contact(contact_defs.ground_align))


if __name__ == "__main__":
    test_refine_motion()

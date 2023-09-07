import numpy as np
from pydrake.all import RigidTransform

import components
import utils
import visualize
from belief import belief_state, dynamics
from simulation import ik_solver


def init():
    X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, 0.3], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([0.0, 0.0, 0.155], [0, 0, 0])
    X_WO = utils.xyz_rpy_deg([0.5, 0, 0.075], [0, 0, 0])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p_0 = belief_state.Particle(
        q_r_0, X_GM, X_WO, "assets/chamfered_hole.sdf", "assets/peg.urdf"
    )
    return p_0


def test_simulate():
    p_0 = init()
    X_WG_d = utils.xyz_rpy_deg([0.5, 0.0, 0.2], [180, 0, 0])
    u_0 = components.CompliantMotion(RigidTransform(), X_WG_d, components.stiff)
    p_1 = dynamics.simulate(p_0, u_0, vis=True)
    print(f"{p_1.sdf=}")


def test_motion_set():
    p0 = init()
    zs = np.linspace(0.2, 0.3, 5)
    rts = [utils.xyz_rpy_deg([0.5, 0.0, z], [180, 0, 0]) for z in zs]
    U = [
        components.CompliantMotion(RigidTransform(), rt, components.stiff) for rt in rts
    ]
    result_set = dynamics.f_cspace(p0, U)
    for i, pT in enumerate(result_set):
        print(f"particle {i}: {pT.contacts}")


def test_belief_dynamics():
    p0 = init()
    grasps = [
        components.Grasp(x=0, z=0.155, pitch=0),
        components.Grasp(x=0, z=0.154, pitch=0),
        components.Grasp(x=0, z=0.156, pitch=0),
    ]
    bin_poses = [
        components.ObjectPose(x=0.5, y=0, yaw=0),
        components.ObjectPose(x=0.5, y=0, yaw=0),
        components.ObjectPose(x=0.5, y=0, yaw=0),
    ]

    b0 = belief_state.Belief.make_particles(grasps, bin_poses, p0)
    X_WG_d = utils.xyz_rpy_deg([0.5, 0.0, 0.2], [180, 0, 0])
    u0 = components.CompliantMotion(RigidTransform(), X_WG_d, components.stiff)
    b1 = dynamics.f_bel(b0, u0)
    print([p.contacts for p in b1.particles])


def test_vis():
    p0 = init()
    p0.q_r = np.array(
        [
            0.19045906,
            1.53978349,
            -0.14104434,
            -0.0698,
            -0.20832431,
            0.66944977,
            0.3623883,
            0.0,
            0.0,
        ]
    )
    p0.env_geom = "assets/empty_world.sdf"
    im = visualize.generate_particle_picture(p0)
    im.save("test.jpg")


if __name__ == "__main__":
    test_vis()
    # test_simulate()

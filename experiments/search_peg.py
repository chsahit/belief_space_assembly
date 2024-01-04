import numpy as np
from pydrake.all import RigidTransform

import components
import dynamics
import state
import utils
import visualize
from planning import randomized_search, refine_motion
from simulation import diagram_factory, ik_solver


def init(
    X_GM_x: float = 0.0, X_GM_z: float = 0.0, pitch: float = 0.01
) -> state.Particle:
    z = 0.155 + X_GM_z
    X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, 0.36], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([X_GM_x, 0.0, z], [180, pitch, 0])
    X_WO = utils.xyz_rpy_deg([0.5, 0, 0.075], [0, 0, 0])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p0 = state.Particle(
        q_r_0,
        X_GM,
        X_WO,
        "assets/big_chamfered_hole.sdf",
        "assets/peg.urdf",
        mu=0.3,
    )
    return p0


def simple_down():
    bottom_faces = (
        ("bin_model::bottom_top", "block::Box_bottom"),
        ("bin_model::front_front", "block::Box_back"),
    )
    bottom_faces = frozenset(bottom_faces)
    chamfer_touch_2 = (
        ("bin_model::front_chamfer_inside", "block::Box_bottom"),
        ("bin_model::front_chamfer_inside", "block::Box_back"),
    )
    front_faces = (
        ("bin_model::front_front", "block::Box_bottom"),
        ("bin_model::front_front", "block::Box_back"),
        ("bin_model::left_left", "block::Box_right"),
    )
    front_faces = frozenset(front_faces)
    chamfer_touch = frozenset(chamfer_touch_2)

    modes = [chamfer_touch_2, front_faces, bottom_faces]
    p0 = init(pitch=-3)
    p1 = init(pitch=3)
    b = state.Belief([p0, p1])
    # diagram_factory.initialize_factory(b.particles)
    traj, tet, st = refine_motion.refine_two_particles(b, modes, max_attempts=5)
    if traj is not None:
        visualize.play_motions_on_belief(
            state.Belief([p0, p1]), traj, fname="four_deg_mu_33.html"
        )
        utils.dump_traj(traj, fname="rot_uncertain.pkl")
    print(f"{tet=}, {st=}")
    print(f"{tet-st=}")
    input()


if __name__ == "__main__":
    simple_down()

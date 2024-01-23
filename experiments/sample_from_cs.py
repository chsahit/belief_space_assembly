import numpy as np
from pydrake.all import RigidTransform

import components
import contact_defs
import dynamics
import puzzle_contact_defs
import state
import utils
import visualize
from planning import search
from simulation import generate_contact_set, ik_solver


def init(X_GM_x: float = 0.0) -> state.Particle:
    X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, 0.3], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([X_GM_x, 0.0, 0.09], [180, 0, 0])
    X_WO = utils.xyz_rpy_deg([0.5, 0, 0.01], [0, 0, 0])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p0 = state.Particle(
        q_r_0,
        X_GM,
        X_WO,
        "assets/big_fixed_puzzle.sdf",
        "assets/moving_puzzle.sdf",
        mu=0.6,
    )
    return p0


def init_pih(X_GM_x: float = 0.0, X_GM_z: float = 0.0) -> state.Particle:
    z = 0.155 + X_GM_z
    X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, 0.36], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([X_GM_x, 0.0, z], [180, 0, 0])
    X_WO = utils.xyz_rpy_deg([0.5, 0, 0.075], [0, 0, 0])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p0 = state.Particle(
        q_r_0,
        X_GM,
        X_WO,
        "assets/big_chamfered_hole.sdf",
        "assets/peg.urdf",
        mu=0.0,
    )
    return p0


def ts2():
    p = init_pih()
    chamfer_touch = (("bin_model::front_chamfer_inside", "block::Box_bottom"),)
    """
    chamfer_touch_2 = (
        ("bin_model::back_chamfer_inside", "block::Box_bottom"),
        ("bin_model::back_chamfer_inside", "block::Box_front"),
    )
    """
    chamfer_touch_2 = (
        ("bin_model::front_chamfer_inside", "block::Box_bottom"),
        ("bin_model::front_chamfer_inside", "block::Box_back"),
        ("bin_model::left_chamfer_inside", "block::Box_bottom"),
        ("bin_model::left_chamfer_inside", "block::Box_right"),
    )
    chamfer_touch = frozenset(chamfer_touch_2)
    front_faces = (
        ("bin_model::back_back", "block::Box_bottom"),
        ("bin_model::back_back", "block::Box_front"),
        ("bin_model::left_left", "block::Box_right"),
    )
    front_faces = frozenset(front_faces)

    X_WGs = generate_contact_set.project_manipuland_to_contacts(
        p, front_faces, num_samples=6
    )
    for X_WG in X_WGs:
        q_r = ik_solver.gripper_to_joint_states(X_WG)
        new_p = p.deepcopy()
        new_p.q_r = q_r
        visualize.show_particle(new_p)
    print("Done")
    input()


def test_ik():
    p = init()
    X_WG = ik_solver.project_manipuland_to_contacts(p, ft)
    q_r = ik_solver.gripper_to_joint_states(X_WG)
    new_p = p.deepcopy()
    new_p.q_r = q_r
    visualize.show_particle(new_p)
    input()


if __name__ == "__main__":
    # ts2()
    # print("\n")
    p = init()
    X_WGs = generate_contact_set.project_manipuland_to_contacts(
        p, puzzle_contact_defs.goal
    )
    q_r = ik_solver.gripper_to_joint_states(X_WGs[0])
    new_p = p.deepcopy()
    new_p.q_r = q_r
    cm = components.CompliantMotion(
        RigidTransform(),
        utils.xyz_rpy_deg([0.5, 0.0, 0.3], [180, 0, 0]),
        components.stiff,
    )
    dynamics.simulate(new_p, cm, vis=True)
    input()

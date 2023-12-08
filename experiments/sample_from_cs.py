import numpy as np
from pydrake.all import RigidTransform

import components
import dynamics
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


top_touch = set((("big_fixed_puzzle::b3", "block::201"),))
ft = set(
    (
        ("big_fixed_puzzle::b3", "block::300"),
        ("big_fixed_puzzle::b3", "block::302"),
    )
)


def test_sampler():
    p0 = init()
    sample = *generate_contact_set.compute_samples_from_contact_set(p0, top_touch)
    R_WM = p0.X_WM.rotation()
    X_WM = RigidTransform(R_WM, sample)
    # TODO: rotation of polyhedron might be funky, since its based around body frame
    X_WG = X_WM.multiply(p.X_GM.inverse())
    q_r = ik_solver.gripper_to_joint_states(X_WG)
    new_p = p.deepcopy()
    new_p.q_r = q_r
    visualize.show_particle(new_p)
    input()


if __name__ == "__main__":
    test_sampler()

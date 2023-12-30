import numpy as np
from pydrake.all import RigidTransform

import components
import dynamics
import state
import utils
import visualize
from planning import randomized_search
from simulation import diagram_factory, ik_solver


def init(X_GM_x: float = 0.0, X_GM_z: float = 0.0) -> state.Particle:
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


def simple_down():
    bottom_faces = (("bin_model::bottom_top", "block::Box_bottom"),)
    modes = [frozenset(bottom_faces)]
    p0 = init()
    b = state.Belief([p0])
    diagram_factory.initialize_factory(b.particles)
    randomized_search.refine_b(b, modes[0])


# TODO: nested refine logic should be refactored into planner/

if __name__ == "__main__":
    simple_down()

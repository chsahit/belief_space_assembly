import numpy as np
from pydrake.all import RigidTransform

import components
import dynamics
import puzzle_contact_defs
import state
import utils
import visualize
from experiments import init_particle
from planning import randomized_search, refine_motion
from simulation import diagram_factory


def simple_down():
    modes = [puzzle_contact_defs.top_touch2, puzzle_contact_defs.bt]
    p0 = init_particle.init_puzzle()
    p1 = init_particle.init_puzzle()
    b = state.Belief([p0, p1])
    result = refine_motion.randomized_refine(b, modes, max_attempts=1)
    if result.traj is not None:
        print("visualize")
        visualize.play_motions_on_belief(
            state.Belief([p0, p1]), result.traj, fname="four_deg_mu_33.html"
        )
        utils.dump_traj(p1.q_r, result.traj, fname="rot_uncertain.pkl")
    print(f"elapsed time: {result.total_time}")
    input()


if __name__ == "__main__":
    simple_down()

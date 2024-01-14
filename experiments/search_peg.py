import numpy as np
from pydrake.all import RigidTransform

import components
import contact_defs
import dynamics
import state
import utils
import visualize
from experiments import init_particle
from planning import randomized_search, refine_motion
from simulation import diagram_factory


def simple_down():
    modes = [
        contact_defs.chamfer_touch_2,
        contact_defs.front_faces,
        contact_defs.bottom_faces_3,
        contact_defs.bottom_faces,
    ]
    p0 = init_particle.init_peg(pitch=-3)
    p1 = init_particle.init_peg(pitch=3)
    b = state.Belief([p0, p1])
    # diagram_factory.initialize_factory(b.particles)
    result = refine_motion.refine_two_particles(b, modes, max_attempts=100)
    if result is not None:
        """
        visualize.play_motions_on_belief(
            state.Belief([p0, p1]), traj, fname="four_deg_mu_33.html"
        )
        """
        utils.dump_traj(p1.q_r, result.traj, fname="rot_uncertain.pkl")
    print(f"elapsed time: {result.total_time}")
    input()


if __name__ == "__main__":
    simple_down()

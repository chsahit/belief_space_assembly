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
    p0 = init_particle.init_peg(pitch=-1)
    p1 = init_particle.init_peg(pitch=1)
    b = state.Belief([p0, p1])
    # diagram_factory.initialize_factory(b.particles)
    traj, tet, st = refine_motion.refine_two_particles(b, modes, max_attempts=100)
    if traj is not None:
        """
        visualize.play_motions_on_belief(
            state.Belief([p0, p1]), traj, fname="four_deg_mu_33.html"
        )
        """
        utils.dump_traj(traj, fname="rot_uncertain.pkl")
    print(f"{tet=}, {st=}")
    print(f"{tet-st=}")
    input()


if __name__ == "__main__":
    simple_down()

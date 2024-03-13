import numpy as np
from pydrake.all import RigidTransform

import components
import contact_defs
import dynamics
import naive_cspace
import state
import utils
import visualize
from experiments import init_particle
from planning import ao_b_est, cobs, randomized_search, refine_motion
from simulation import diagram_factory


def auto_tp_sd():
    p0 = init_particle.init_peg(pitch=-3)
    p1 = init_particle.init_peg(pitch=0)
    p2 = init_particle.init_peg(pitch=3)
    b = state.Belief([p0, p1, p2])
    result = cobs.cobs(b, contact_defs.bottom_faces_2)
    # result = ao_b_est.b_est(b, contact_defs.bottom_faces_2)
    if result.traj is not None:
        visualize.play_motions_on_belief(state.Belief([p0, p1, p2]), result.traj)
        input()


def simple_down():
    p0 = init_particle.init_peg(y=-0.03)
    p1 = init_particle.init_peg(pitch=0)
    p2 = init_particle.init_peg(y=0.03)
    b = state.Belief([p0, p1, p2])
    modes = [
        contact_defs.chamfer_touch_2,
        contact_defs.front_faces,
        contact_defs.bottom_faces_fully_constrained,
    ]
    # diagram_factory.initialize_factory(b.particles)
    result = refine_motion.randomized_refine(b, modes, max_attempts=10)
    if result.traj is not None:
        print("visualize")
        try:
            visualize.play_motions_on_belief(
                state.Belief([p0, p1, p2]), result.traj, fname="four_deg_mu_33.html"
            )
            utils.dump_traj(p1.q_r, result.traj, fname="rot_uncertain.pkl")
        except Exception as e:
            print(e)
    print(f"elapsed time: {result.total_time}")
    input()


if __name__ == "__main__":
    auto_tp_sd()

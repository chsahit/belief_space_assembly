import itertools
import pickle

import contact_defs
import state
from experiments import init_particle
from planning import refine_motion


def sweep_pitches():
    schedule = [
        contact_defs.chamfer_touch_2,
        contact_defs.front_faces,
        contact_defs.bottom_faces_3,
        contact_defs.bottom_faces,
    ]
    results = dict()
    deviations = [1, 2, 3, 4, 5]
    compliance_search = [True, False]
    experiment_params = itertools.product(deviations, compliance_search)
    for deviation, do_compliance in experiment_params:
        print(f"{deviation=}, {do_compliance=}")
        p0 = init_particle.init_peg(pitch=-deviation)
        p1 = init_particle.init_peg(pitch=deviation)
        b = state.Belief([p0, p1])
        experiment_label = str(deviation) + "_" + str(compliance_search)
        results[experiment_label] = refine_motion.refine_two_particles(
            b, schedule, search_compliance=do_compliance, max_attempts=5
        )
        print(str(results[experiment_label]) + "\n")
    with open("sweep_rsults.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    sweep_pitches()

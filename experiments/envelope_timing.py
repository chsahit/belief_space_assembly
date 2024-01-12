import itertools
import pickle

import contact_defs
import puzzle_contact_defs
import state
from experiments import init_particle
from planning import refine_motion

pitch_sweep_peg = ("pitch", [1, 2, 3, 4, 5], "peg")
x_sweep_puzzle = ("X_GM_x", [0.001, 0.003, 0.005, 0.007, 0.009], "puzzle")
peg_schedule = [
    contact_defs.chamfer_touch_2,
    contact_defs.front_faces,
    contact_defs.bottom_faces_3,
    contact_defs.bottom_faces,
]
puzzle_schedule = [
    puzzle_contact_defs.top_touch2,
    puzzle_contact_defs.bt,
    puzzle_contact_defs.bottom,
    puzzle_contact_defs.goal,
]


def sweep(dof, deviations, geometry, schedule):
    if geometry == "peg":
        initializer = init_particle.init_peg
    elif geometry == "puzzle":
        initializer = init_particle.init_puzzle
    else:
        raise NotImplementedError
    results = dict()
    compliance_search = [True, False]
    experiment_params = itertools.product(deviations, compliance_search)
    for deviation, do_compliance in experiment_params:
        print(f"{deviation=}, {do_compliance=}")
        kwarg_0 = {dof: -deviation, "mu": 0.0}
        kwarg_1 = {dof: deviation, "mu": 0.0}
        p0 = initializer(**kwarg_0)
        p1 = initializer(**kwarg_1)
        b = state.Belief([p0, p1])
        experiment_label = str(deviation) + "_" + str(compliance_search)
        results[experiment_label] = refine_motion.refine_two_particles(
            b, schedule, search_compliance=do_compliance, max_attempts=5
        )
        print(str(results[experiment_label]) + "\n")
    with open("sweep_rsults.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    # sweep(*pitch_sweep_peg, peg_schedule)
    sweep(*x_sweep_puzzle, puzzle_schedule)

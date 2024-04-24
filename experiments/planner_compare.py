import itertools
import pickle

import contact_defs
import puzzle_contact_defs
import state
import visualize
from experiments import init_particle
from planning import ao_b_est, cobs

pitch_sweep_peg = ("pitch", [1, 3, 5, 7, 9, 12], "peg")
pitch_sweep_peg = ("pitch", [0.5, 1, 1.5, 2], "peg")
pitch_sweep_puzzle = ("pitch", [1.5, 2, 3, 3.5, 4], "puzzle")
x_sweep_peg = ("X_GM_x", [0.02, 0.04, 0.06], "peg")
z_sweep_puzzle = ("X_GM_x", [0.0025, 0.005, 0.01, 0.015, 0.02], "puzzle")
z_sweep_peg = ("X_GM_z", [0.005, 0.01, 0.015, 0.02], "peg")
y_sweep_peg = ("y", [0.01, 0.02, 0.03], "peg")

planners = {
    "b_est": ao_b_est.b_est,
    "cobs": cobs.cobs,
    "no_k": cobs.no_k,
    "no_gp": cobs.no_gp,
    "no_replan": cobs.no_replan,
}


def sweep(dof, deviations, geometry):
    print(f"{geometry=}, {dof=}")
    if geometry == "peg":
        initializer = init_particle.init_peg
        goal = contact_defs.bottom_faces_2
    elif geometry == "puzzle":
        initializer = init_particle.init_puzzle
        goal = puzzle_contact_defs.side
    else:
        raise NotImplementedError
    results = dict()
    experiment_params = itertools.product(deviations, list(planners.keys()))
    for deviation, planner in experiment_params:
        print(f"{deviation=}, {planner=}")
        kwarg_0 = {dof: -deviation}
        kwarg_1 = {dof: 0}
        kwarg_2 = {dof: deviation}
        p0 = initializer(**kwarg_0)
        p1 = initializer(**kwarg_1)
        p2 = initializer(**kwarg_2)
        b = state.Belief([p0, p1, p2])
        experiment_label = (str(deviation), str(planner))
        trials = 3
        experiment_results = []
        for trial_idx in range(trials):
            print(f"TRIAL: {trial_idx}")
            try:
                plan_result = planners[planner](b, goal)
                experiment_results.append(plan_result)
                print(str(experiment_results[-1]))
            except Exception as e:
                print(f"planner_compare.py {e=}")
        print("\n")
        results[experiment_label] = experiment_results
    fname = dof + "_" + geometry + "_" + "sweep_results.pkl"
    with open(fname, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    # sweep(*pitch_sweep_peg)
    visualize.show_benchmarks("pitch_peg_sweep_results.pkl")

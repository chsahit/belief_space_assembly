import itertools
import traceback

import contact_defs
import counters
import puzzle_contact_defs
import state
import utils
import visualize
from experiments import init_particle
from planning import ao_b_est, cobs

pitch_sweep_peg = ("pitch", [1, 2, 3, 4], "peg")
x_sweep_peg = ("X_GM_x", [0.01, 0.02], "peg")
y_sweep_peg = ("y", [0.01, 0.02], "peg")

pitch_sweep_puzzle = ("pitch", [1, 2], "puzzle")
x_sweep_puzzle = ("X_GM_x", [0.01, 0.02], "puzzle")
y_sweep_puzzle = ("y", [0.01], "puzzle")
z_sweep_puzzle = ("X_GM_z", [0.0025, 0.005, 0.01, 0.015, 0.02], "puzzle")
z_sweep_peg = ("X_GM_z", [0.005, 0.01, 0.015, 0.02], "peg")

planners = {
    # "cobs": cobs.cobs,
    # "no_k": cobs.no_k,
    # "b_est": ao_b_est.b_est,
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
    experiment_params = itertools.product(deviations, list(planners.keys()))
    fname = dof + "_" + geometry + "_" + "sweep_results.pkl"
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
        trials = 10
        experiment_results = []
        for trial_idx in range(trials):
            print(f"TRIAL: {trial_idx}")
            counters.reset_time()
            counters.reset_posteriors()
            try:
                plan_result = planners[planner](b, goal)
                plan_result.num_posteriors = counters.get_posterior_count()
                plan_result.sim_time = counters.get_time()
                experiment_results.append(plan_result)
                print(str(experiment_results[-1]))
            except Exception as e:
                print(f"planner_compare.py {e=}")
                traceback.print_exception(type(e), e, e.__traceback__)
        utils.log_experiment_result(fname, experiment_label, experiment_results)
        del experiment_results
        print("\n")


if __name__ == "__main__":
    """
    sweep(*pitch_sweep_peg)
    sweep(*x_sweep_peg)
    sweep(*y_sweep_peg)
    sweep(*pitch_sweep_puzzle)
    sweep(*y_sweep_puzzle)
    """
    # sweep(*x_sweep_puzzle)
    visualize.show_benchmarks("pitch_puzzle_sweep_results.pkl")
    visualize.show_benchmarks("X_GM_x_puzzle_sweep_results.pkl")
    visualize.show_benchmarks("y_puzzle_sweep_results.pkl")

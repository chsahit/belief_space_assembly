import itertools
import pickle

import contact_defs
import puzzle_contact_defs
import state
import visualize
from experiments import init_particle
from planning import refine_motion

pitch_sweep_peg = ("pitch", [4, 4.5, 5], "peg")
pitch_sweep_puzzle = ("pitch", [1, 1.5, 2, 2.5, 3, 3.5, 4], "puzzle")
x_sweep_puzzle = ("X_GM_x", [0.001, 0.003, 0.005, 0.007, 0.009], "puzzle")
z_sweep_peg = ("X_GM_z", [0.005, 0.01, 0.015], "peg")
# x_sweep_peg = ("X_GM_x", [0.0075, 0.01, 0.0125, 0.015], "peg")
x_sweep_peg = ("X_GM_x", [0.0175, 0.02, 0.0225], "peg")
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
    print(f"{geometry=}, {dof=}")
    if geometry == "peg":
        initializer = init_particle.init_peg
    elif geometry == "puzzle":
        initializer = init_particle.init_puzzle
    else:
        raise NotImplementedError
    results = dict()
    compliance_gp = [(True, True), (False, True), (True, False)]
    experiment_params = itertools.product(deviations, compliance_gp)
    stopped_params = []
    for deviation, (do_compliance, do_gp) in experiment_params:
        print(f"{deviation=}, {do_compliance=} {do_gp=}")
        if (do_compliance, do_gp) in stopped_params:
            print("not running experiment because envelope is smaller\n")
        kwarg_0 = {dof: -deviation}
        kwarg_1 = {dof: deviation}
        kwarg_2 = {dof: 0}
        p0 = initializer(**kwarg_0)
        p1 = initializer(**kwarg_1)
        p2 = initializer(**kwarg_2)
        b = state.Belief([p0, p1, p2])
        experiment_label = (str(deviation), str(do_compliance), str(do_gp))
        trials = 4
        experiment_results = []
        for trial_idx in range(trials):
            print(f"TRIAL: {trial_idx}")
            experiment_results.append(
                refine_motion.randomized_refine(
                    b,
                    schedule,
                    search_compliance=do_compliance,
                    do_gp=do_gp,
                    max_attempts=10,
                )
            )
            if experiment_results[-1].traj is not None:
                break
            print(str(experiment_results[-1]))
        print("\n")
        if all([result.traj is None for result in experiment_results]):
            print(f"stopping ({do_compliance=}, {do_gp=})")
            stopped_params.append((do_compliance, do_gp))
        results[experiment_label] = experiment_results
    fname = dof + "_" + geometry + "_" + "sweep_results.pkl"
    with open(fname, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    # visualize.show_planning_results("pitch_peg_sweep_results.pkl")
    # sweep(*pitch_sweep_puzzle, puzzle_schedule)
    # sweep(*pitch_sweep_peg, peg_schedule)
    # sweep(*x_sweep_puzzle, puzzle_schedule)
    # sweep(*x_sweep_peg, peg_schedule)
    sweep(*z_sweep_peg, peg_schedule)

    # b = state.Belief([init_particle.init_peg(-0.009), init_particle.init_peg(), init_particle.init_peg(0.009)])
    # visualize.playback_result(b, "X_GM_x_peg_sweep_results.pkl")

    # b = state.Belief([init_particle.init_peg(pitch=-4), init_particle.init_peg(), init_particle.init_peg(pitch=4)])
    # visualize.playback_result(b, "pitch_peg_sweep_results.pkl")

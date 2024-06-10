import contact_defs
import state
import utils
from experiments import init_particle
from planning import ao_b_est, cobs


def make_peg_plan(fname: str, planner):
    # p0 = init_particle.init_peg(y=-0.015)
    p0 = init_particle.init_peg(y=-0.0125, X_GM_x=-0.005, mu=0.5)
    p1 = init_particle.init_peg(pitch=0, mu=0.5)
    p2 = init_particle.init_peg(y=0.0125, X_GM_x=0.005, mu=0.5)
    b = state.Belief([p0, p1, p2])
    result = planner(b, contact_defs.peg_goal)
    if result.traj is not None:
        utils.pickle_trajectory(p1, result.traj, fname=fname)
        # utils.log_experiment_result("logs/peg_soln.pkl", "result", [result])


def make_plans():
    baseline = [(ao_b_est.b_est, "b_est")]
    method = [(cobs.cobs, "cobs"), (cobs.no_k, "no_k")]
    del method
    for planner, pname in baseline:
        for i in range(10):
            fname = f"logs/{pname}_soln_{i}.pkl"
            make_peg_plan(fname, planner)
            print(f"logged: {fname}")


if __name__ == "__main__":
    make_plans()

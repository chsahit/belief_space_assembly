import pickle

import contact_defs
import state
import utils
import visualize
from experiments import init_particle
from planning import ao_b_est, cobs


def auto_tp_sd(ours: bool = True):
    # p0 = init_particle.init_peg(y=-0.015)
    p0 = init_particle.init_peg(y=-0.01, pitch=-2)
    p1 = init_particle.init_peg(pitch=0)
    p2 = init_particle.init_peg(y=0.01, pitch=2)
    b = state.Belief([p0, p1, p2])
    if ours:
        result = cobs.cobs(b, contact_defs.peg_goal, log_samples=False)
        assert result.traj is not None
        # utils.pickle_trajectory(p1, result.traj)
    else:
        result = ao_b_est.b_est(b, contact_defs.peg_goal)
    if result.traj is not None:
        # utils.log_experiment_result("logs/peg_soln.pkl", "result", [result])
        visualize.play_motions_on_belief(state.Belief([p0, p1, p2]), result.traj)
        input()


def show_planner_trace():
    visualize.show_belief_space_traj("logs/samples.pkl")


def view_peg_soln():
    p0 = init_particle.init_peg(y=-0.01)
    p1 = init_particle.init_peg(pitch=0.0)
    p2 = init_particle.init_peg(y=0.01)
    b = state.Belief([p0, p1, p2])
    with open("logs/puzzle_soln.pkl", "rb") as f:
        result = pickle.load(f)
    traj = result["result"][0].traj
    visualize.play_motions_on_belief(b, traj, pretty=True)
    input()


if __name__ == "__main__":
    # view_peg_soln()
    auto_tp_sd(ours=True)
    # show_planner_trace()

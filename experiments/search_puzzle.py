import logging
import pickle

import contact_defs
import state
import utils
import visualize
from experiments import init_particle
from planning import ao_b_est, cobs, refine_motion

logging.basicConfig(level=logging.INFO)


def run_search(ours: bool = True):
    p0 = init_particle.init_puzzle(X_GM_x=-0.01)
    p1 = init_particle.init_puzzle(pitch=0.0)
    p2 = init_particle.init_puzzle(X_GM_x=0.01)
    b = state.Belief([p0, p1, p2])
    if ours:
        result = cobs.cobs(b, contact_defs.puzzle_goal)
    else:
        result = ao_b_est.b_est(b, contact_defs.puzzle_goal)
    if result.traj is not None:
        print("vis")
        utils.log_experiment_result("puzzle_soln.pkl", "result", [result])
        visualize.play_motions_on_belief(
            state.Belief([p0, p1, p2]),
            result.traj,
            fname="puzzle_soln.html",
            pretty=True,
        )
        input()
    else:
        logging.error("no soln found")


def view_puzzle_soln():
    p0 = init_particle.init_puzzle(X_GM_x=-0.01)
    p1 = init_particle.init_puzzle(pitch=0.0)
    p2 = init_particle.init_puzzle(X_GM_x=0.01)
    b = state.Belief([p0, p1, p2])
    with open("puzzle_soln.pkl", "rb") as f:
        result = pickle.load(f)
    traj = result["result"][0].traj
    visualize.play_motions_on_belief(b, traj, pretty=True)
    input()


if __name__ == "__main__":
    view_puzzle_soln()

import puzzle_contact_defs
import state
import utils
import visualize
from experiments import init_particle
from planning import ao_b_est, cobs, refine_motion


def run_search(ours: bool = True):
    p0 = init_particle.init_puzzle(pitch=2)
    p1 = init_particle.init_puzzle(pitch=0.0)
    p2 = init_particle.init_puzzle(pitch=-2)
    b = state.Belief([p0, p1, p2])
    if ours:
        result = cobs.cobs(b, puzzle_contact_defs.side)
    else:
        result = ao_b_est.b_est(b, puzzle_contact_defs.side)
    if result.traj is not None:
        visualize.play_motions_on_belief(
            state.Belief([p0, p1, p2]), result.traj, fname="puzzle_soln.html"
        )
        input()


def simple_down():
    modes = [
        puzzle_contact_defs.top_touch2,
        puzzle_contact_defs.bt,
        puzzle_contact_defs.bottom,
        puzzle_contact_defs.goal,
    ]
    p0 = init_particle.init_puzzle(pitch=2.0)
    p1 = init_particle.init_puzzle(pitch=-2.0)
    b = state.Belief([p0, p1])
    result = refine_motion.randomized_refine(b, modes, max_attempts=10)
    if result.traj is not None:
        print("visualize")
        visualize.play_motions_on_belief(
            state.Belief([p0, p1]), result.traj, fname="four_deg_mu_33.html"
        )
        utils.dump_traj(p1.q_r, result.traj, fname="rot_uncertain.pkl")
    print(f"elapsed time: {result.total_time}")
    input()


if __name__ == "__main__":
    run_search(ours=True)

import contact_defs
import state
import utils
import visualize
from experiments import init_particle
from planning import ao_b_est, cobs, refine_motion
from simulation import ik_solver


def auto_tp_sd(ours: bool = True):
    # p0 = init_particle.init_peg(y=-0.015)
    p0 = init_particle.init_peg(pitch=-2)
    p1 = init_particle.init_peg(pitch=0)
    p2 = init_particle.init_peg(pitch=2)
    b = state.Belief([p0, p1, p2])
    if ours:
        # result = cobs.cobs(b, contact_defs.bottom_faces_2)
        result = cobs.cobs(b, contact_defs.bottom_faces_2, log_samples=True)
        for u in result.traj:
            ik_solver.update_motion_qd(u)
        utils.pickle_trajectory(p1, result.traj)
    else:
        result = ao_b_est.b_est(b, contact_defs.bottom_faces_2)
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


def show_planner_trace():
    visualize.show_belief_space_traj("samples.pkl")


if __name__ == "__main__":
    # auto_tp_sd(ours=True)
    show_planner_trace()

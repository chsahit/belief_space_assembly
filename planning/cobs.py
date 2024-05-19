import pickle
import time
from typing import List, Tuple

import components
import contact_defs
import cspace
import dynamics
import sampler
import state
import visualize
from planning import refine_motion
from simulation import ik_solver


def show_task_plan(p_repr: state.Particle, task_plan: List[components.ContactState]):
    nominal_poses = []
    for step in task_plan[1:]:
        X_WG = sampler.sample_from_contact(p_repr, step, 1)
        q_r = ik_solver.gripper_to_joint_states(X_WG[0])
        nominal_poses.append((X_WG, q_r))
    sample_particles = []
    for pose in nominal_poses:
        p_pose = p_repr.deepcopy()
        p_pose.q_r = pose[1]
        sample_particles.append(p_pose)
    for i, particle in enumerate(sample_particles):
        print(f"{task_plan[i+1]=}")
        visualize.show_particle(particle)
    breakpoint()


def prune_edge(
    graph_init: components.TaskGraph,
    lr: Tuple[components.ContactState, components.ContactState],
) -> components.TaskGraph:
    if str(lr[0]) < str(lr[1]):
        lr = (lr[1], lr[0])
    graph_init.E.remove(lr)
    return graph_init


def cobs(
    b0: state.Belief,
    goal: components.ContactState,
    opt_compliance: bool = True,
    do_gp: bool = True,
    do_replan: bool = True,
    timeout: float = 1000,
    log_samples: bool = False,
) -> components.PlanningResult:
    start_time = time.time()
    p_repr = b0.particles[1]
    p_repr._update_contact_data()
    cspace_slice = cspace.ConstructCspaceSlice(
        cspace.ConstructEnv(p_repr), p_repr.X_WM.rotation()
    )
    graph = cspace.label_mesh(cspace_slice, p_repr)
    max_tp_attempts = 50
    attempt_samples = dict()
    for tp_attempt in range(max_tp_attempts):
        goal_achieved = False
        refine_from = contact_defs.fs
        b_curr = b0
        trajectory = []
        start_pose = b0.mean().X_WM.translation()
        contact_seq = []
        step = 0
        if not do_replan:
            task_plan = cspace.make_task_plan(
                graph, refine_from, goal, b_curr.direction(), start_pose=start_pose
            )
        if time.time() - start_time > timeout:
            break
        while not goal_achieved:
            if do_replan:
                nominal_plan = cspace.make_task_plan(
                    graph, refine_from, goal, b_curr.direction(), start_pose=start_pose
                )
            else:
                nominal_plan = task_plan[step:]
                step += 1
            start_pose = None
            # show_task_plan(p_repr, nominal_plan)
            contact_seq.append(nominal_plan[1])
            intermediate_result = refine_motion.randomized_refine(
                b_curr,
                [nominal_plan[1]],
                search_compliance=opt_compliance,
                max_attempts=1,
                do_gp=do_gp,
            )
            attempt_samples[nominal_plan[1]] = refine_motion.sample_logs_rr
            refine_motion.sample_logs_rr = []
            if intermediate_result.traj is None:
                lr = list(intermediate_result.last_refined)
                lr[0] = nominal_plan[0]
                graph = prune_edge(graph, tuple(lr))
                break
            trajectory.extend(intermediate_result.traj)
            if nominal_plan[1] == goal:
                goal_achieved = True
                break
            for u in intermediate_result.traj:
                b_curr = dynamics.f_bel(b_curr, u)
            refine_from = nominal_plan[1]
        if goal_achieved:
            if log_samples:
                print(f"{contact_seq=}")
                attempt_samples["trajectory"] = dynamics.sequential_f_bel(
                    b0, trajectory
                )
                attempt_samples["contact_seq"] = contact_seq
                dump_attempt_samples(attempt_samples)
            T = time.time() - start_time
            return components.PlanningResult(trajectory, T, 0, 0, None)
    T = time.time() - start_time
    return components.PlanningResult(None, T, 0, 0, None)


def dump_attempt_samples(attempt_samples):
    for b in attempt_samples["trajectory"]:
        for p in b.particles:
            p.cspace_repr = None
    with open("samples.pkl", "wb") as f:
        pickle.dump(attempt_samples, f)


def no_k(
    b0: state.Belief,
    goal: components.ContactState,
) -> components.PlanningResult:
    return cobs(b0, goal, opt_compliance=False)


def no_gp(b0: state.Belief, goal: components.ContactState) -> components.PlanningResult:
    return cobs(b0, goal, do_gp=False)


def no_replan(
    b0: state.Belief, goal: components.ContactState
) -> components.PlanningResult:
    return cobs(b0, goal, do_replan=False)

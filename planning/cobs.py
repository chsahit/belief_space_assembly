from typing import List, Tuple

import networkx as nx
from pydrake.all import HPolyhedron, RigidTransform

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
        X_WG = sampler.sample_from_contact(p_repr, step[0], step[1], 1)
        q_r = ik_solver.gripper_to_joint_states(X_WG[0])
        nominal_poses.append((X_WG, q_r))
    sample_particles = []
    for pose in nominal_poses:
        p_pose = p_repr.deepcopy()
        p_pose.q_r = pose[1]
        sample_particles.append(p_pose)
    for particle in sample_particles:
        visualize.show_particle(particle)
    breakpoint()


def prune_edge(
    graph_init: cspace.CSpaceGraph,
    lr: Tuple[components.ContactState, components.ContactState],
) -> nx.Graph:
    edges = []
    for e in graph_init.E:
        labels_a = (e[0].label, e[1].label)
        labels_b = (e[1].label, e[0].label)
        delete_edge = labels_a == lr or labels_b == lr
        if not delete_edge:
            edges.append(e)
    graph = cspace.CSpaceGraph(graph_init.V, edges)
    return graph


def cobs(
    b0: state.Belief,
    goal: components.ContactState,
    opt_compliance: bool = True,
    do_gp: bool = True,
) -> components.PlanningResult:
    p_repr = b0.particles[1]
    p_repr._update_contact_data()
    transformed_manip_poly = dict()
    for name, geom in p_repr._manip_poly.items():
        transformed_geom = cspace.TF_HPolyhedron(
            HPolyhedron(*geom), RigidTransform(p_repr.X_WM.rotation())
        )
        transformed_manip_poly[name] = (transformed_geom.A(), transformed_geom.b())
    graph = cspace.MakeModeGraphFromFaces(p_repr.constraints, transformed_manip_poly)
    max_tp_attempts = 15
    t = components.Time(0, 0, 0)
    for tp_attempt in range(max_tp_attempts):
        goal_achieved = False
        refine_from = contact_defs.fs
        b_curr = b0
        trajectory = []
        start_pose = b0.mean().X_WM.translation()
        while not goal_achieved:
            nominal_plan = cspace.make_task_plan(
                graph, refine_from, goal, b_curr.direction(), start_pose=start_pose
            )
            start_pose = None
            print(f"task plan = {nominal_plan}")
            # if refine_from == contact_defs.fs:
            #     show_task_plan(p_repr, nominal_plan)
            intermediate_result = refine_motion.randomized_refine(
                b_curr,
                [nominal_plan[1]],
                search_compliance=opt_compliance,
                max_attempts=1,
                do_gp=do_gp,
            )
            t.add_result(intermediate_result)
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
            return components.PlanningResult(
                trajectory, t.total_time, t.sim_time, t.num_posteriors, None
            )
    return components.PlanningResult(
        None, t.total_time, t.sim_time, t.num_posteriors, None
    )


def no_k(
    b0: state.Belief,
    goal: components.ContactState,
) -> components.PlanningResult:
    return cobs(b0, goal, opt_compliance=False)


def no_gp(b0: state.Belief, goal: components.ContactState) -> components.PlanningResult:
    return cobs(b0, goal, do_gp=False)

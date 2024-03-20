from typing import Tuple

import networkx as nx

import components
import contact_defs
import dynamics
import naive_cspace
import state
from planning import refine_motion


def prune_edge(
    graph_init: naive_cspace.CSpaceGraph,
    lr: Tuple[components.ContactState, components.ContactState],
) -> nx.Graph:
    edges = []
    for e in graph_init.E:
        labels_a = (e[0].label, e[1].label)
        labels_b = (e[1].label, e[0].label)
        delete_edge = labels_a == lr or labels_b == lr
        if not delete_edge:
            edges.append(e)
    graph = naive_cspace.CSpaceGraph(graph_init.V, edges, [])
    return graph


def cobs(
    b0: state.Belief,
    goal: components.ContactState,
    opt_compliance: bool = True,
) -> components.PlanningResult:
    print("warning, nominal particle hardedcoded to i=1")
    p_repr = b0.particles[1]
    p_repr._update_contact_data()
    graph = naive_cspace.MakeModeGraphFromFaces(p_repr.constraints, p_repr._manip_poly)
    max_tp_attempts = 15
    t = components.Time(0, 0, 0)
    vert_cache = []
    validated_cache = set()
    for tp_attempt in range(max_tp_attempts):
        goal_achieved = False
        refine_from = contact_defs.fs
        b_curr = b0
        trajectory = []
        while not goal_achieved:
            configs = [p.X_OM.translation() for p in b_curr.particles]
            nominal_plan = naive_cspace.make_task_plan(
                graph, refine_from, goal, configs
            )
            print(f"task plan = {nominal_plan}")
            intermediate_result = refine_motion.randomized_refine(
                b_curr,
                [nominal_plan[1]],
                search_compliance=opt_compliance,
                max_attempts=1,
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

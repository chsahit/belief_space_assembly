import components
import naive_cspace
import state
from planning import refine_motion


def cobs(
    b0: state.Belief,
    init: components.ContactState,
    goal: components.ContactState,
    opt_compliance: bool = True,
) -> components.PlanningResult:
    print("warning, nominal particle hardedcoded to i=1")
    p_repr = b0.particles[1]
    p_repr._update_contact_data()
    graph = naive_cspace.MakeModeGraphFromFaces(p_repr.constraints, p_repr._manip_poly)
    max_tp_attempts = 15
    total_tet = 0
    total_st = 0
    total_np = 0
    vert_cache = []
    for tp_attempt in range(max_tp_attempts):
        modes = naive_cspace.make_task_plan(graph, init, goal)
        print(f"task plan = {modes}")
        result = refine_motion.randomized_refine(
            b0, modes, search_compliance=opt_compliance, max_attempts=1
        )
        total_tet += result.total_time
        total_st = result.sim_time
        total_np = result.num_posteriors
        if result.traj is not None:
            return components.PlanningResult(
                result.traj, total_tet, total_st, total_np, None
            )
        lr = result.last_refined
        for m in modes:
            if m == lr[1]:
                break
            vert_cache.append(m)
        edges = []
        for e in graph.E:
            if (e[0].label, e[1].label) == lr or (e[1].label, e[0].label) == lr:
                continue
            else:
                edges.append(e)
        # print(f"{vert_cache=}")
        graph = naive_cspace.CSpaceGraph(graph.V, edges, vert_cache)
    return components.PlanningResult(None, total_tet, total_st, total_np, None)

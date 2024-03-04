import components
import naive_cspace
import state
from planning import refine_motion


def cobs(
    b0: state.Belief, init: components.ContactState, goal: components.ContactState
) -> components.PlanningResult:
    print("warning, nominal particle hardedcoded to i=1")
    p_repr = b0.particles[1]
    p_repr._update_contact_data()
    graph = naive_cspace.MakeModeGraphFromFaces(p_repr.constraints, p_repr._manip_poly)
    max_tp_attempts = 5
    for tp_attempt in range(max_tp_attempts):
        modes = naive_cspace.make_task_plan(graph, init, goal)
        print(f"task plan = {modes}")
        result = refine_motion.randomized_refine(b0, modes, max_attempts=1)
        if result.traj is not None:
            return result
        lr = result.last_refined
        edges = []
        for e in graph.E:
            if (e[0].label, e[1].label) == lr or (e[1].label, e[0].label) == lr:
                print(f"deleting {e}")
                continue
            else:
                edges.append(e)
        graph = naive_cspace.CSpaceGraph(graph.V, edges)

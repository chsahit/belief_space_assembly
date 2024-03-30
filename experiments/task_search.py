import naive_cspace

import contact_defs
from experiments import init_particle


def make_cspace_graph():
    p = init_particle.init_peg()
    p._update_contact_data()
    B = naive_cspace.MakeWorkspaceObjectFromFaces(p.constraints)
    A = naive_cspace.MakeWorkspaceObjectFromFaces(p._manip_poly)
    graph = naive_cspace.make_graph([A], [B])
    naive_cspace.render_graph(graph)
    plan = naive_cspace.make_task_plan(
        graph, contact_defs.chamfer_init, contact_defs.bottom_faces_2
    )
    print(plan)


if __name__ == "__main__":
    make_cspace_graph()

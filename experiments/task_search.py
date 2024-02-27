import naive_cspace
from experiments import init_particle


def make_cspace_graph():
    p = init_particle.init_peg()
    p._update_contact_data()
    B = naive_cspace.WorkspaceObject("", None, p.constraints)
    A = naive_cspace.WorkspaceObject("", None, p._manip_poly)
    graph = naive_cspace.make_graph([A], [B])
    print(graph)


if __name__ == "__main__":
    make_cspace_graph()

from pydrake.all import Diagram

from simulation import plant_builder

sim_diagrams = dict()
collision_diagrams = dict()
offset = 0


def initialize_factory(P):
    global sim_diagrams
    global collision_diagrams
    global offset
    for p_idx, p in enumerate(P):
        sim_diagram, _ = plant_builder.make_plant(
            p.q_r, p.X_GM, p.X_WO, p.env_geom, p.manip_geom, mu=p.mu
        )
        c_diagram, _ = plant_builder.make_plant(
            p.q_r,
            p.X_GM,
            p.X_WO,
            p.env_geom,
            p.manip_geom,
            mu=p.mu,
            collision_check=True,
        )
        p._sim_id = p_idx + offset
        sim_diagrams[p._sim_id] = sim_diagram
        collision_diagrams[p._sim_id] = c_diagram

    offset += len(P)

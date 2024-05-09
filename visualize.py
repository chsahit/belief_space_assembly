import pickle
from collections import defaultdict
from typing import List

import networkx as nx
import plotly.graph_objects as go
from pydrake.all import (
    CollisionFilterDeclaration,
    DiagramBuilder,
    Meshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    RigidTransform,
    Role,
    RoleAssign,
    Simulator,
)

import components
import dynamics
import state
import utils
from simulation import ik_solver, plant_builder


# yoinked from https://github.com/mpetersen94/gcs/blob/main/reproduction/prm_comparison/helpers.py
def set_transparency_of_models(plant, model_instances, color, alpha, scene_graph):
    """Sets the transparency of the given models."""
    inspector = scene_graph.model_inspector()
    for model in model_instances:
        for body_id in plant.GetBodyIndices(model):
            frame_id = plant.GetBodyFrameIdOrThrow(body_id)
            for geometry_id in inspector.GetGeometries(frame_id, Role.kIllustration):
                properties = inspector.GetIllustrationProperties(geometry_id)
                try:
                    phong = properties.GetProperty("phong", "diffuse")
                    if color is not None:
                        phong.set(*color, alpha)
                    else:
                        phong.set(phong.r(), phong.g(), phong.b(), alpha)
                    properties.UpdateProperty("phong", "diffuse", phong)
                    scene_graph.AssignRole(
                        plant.get_source_id(),
                        geometry_id,
                        properties,
                        RoleAssign.kReplace,
                    )
                except Exception as e:
                    pass


def _make_combined_plant(b: state.Belief, meshcat: Meshcat):
    builder = DiagramBuilder()
    plant, scene_graph, parser = plant_builder.init_plant(builder, timestep=0.005)
    instance_list = list()

    for i, p in enumerate(b.particles):
        panda = parser.AddModels("assets/panda_arm_hand.urdf")[0]
        plant.RenameModelInstance(panda, "panda_" + str(i))
        env_geometry = parser.AddModels(p.env_geom)[0]
        plant.RenameModelInstance(env_geometry, "obj_" + str(i))
        manipuland = parser.AddModels(p.manip_geom)[0]
        plant.RenameModelInstance(manipuland, "block_" + str(i))
        plant_builder._weld_geometries(
            plant, p.X_GM, p.X_WO, panda, manipuland, env_geometry
        )
        plant_builder._set_frictions(
            plant, scene_graph, [env_geometry, manipuland], p.mu
        )
        instance_list.append((panda, env_geometry, manipuland))
    plant.Finalize()

    colors = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    for i, p in enumerate(b.particles):
        P, O, M = instance_list[i]
        set_transparency_of_models(plant, [P, O, M], colors[i % 3], 0.5, scene_graph)
        plant.SetDefaultPositions(P, p.q_r)
        plant_builder.wire_controller(
            P,
            f"controller_{str(i)}",
            f"panda_{str(i)}",
            f"setpoint_{str(i)}",
            builder,
            plant,
        )

    meshcat_vis = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat, MeshcatVisualizerParams()
    )
    diagram = builder.Build()
    manager = scene_graph.collision_filter_manager()
    for p_idx_i in range(len(b.particles)):
        P_i, O_i, M_i = instance_list[p_idx_i]
        P_i_bodies = plant.GetBodyIndices(P_i)
        O_i_bodies = plant.GetBodyIndices(O_i)
        M_i_bodies = plant.GetBodyIndices(M_i)
        geom_set_i = plant.CollectRegisteredGeometries(
            [plant.get_body(b_idx) for b_idx in (P_i_bodies + O_i_bodies + M_i_bodies)]
        )
        for p_idx_j in range(p_idx_i + 1, len(b.particles)):
            P_j, O_j, M_j = instance_list[p_idx_j]
            P_j_bodies = plant.GetBodyIndices(P_j)
            O_j_bodies = plant.GetBodyIndices(O_j)
            M_j_bodies = plant.GetBodyIndices(M_j)
            geom_set_j = plant.CollectRegisteredGeometries(
                [
                    plant.get_body(b_idx)
                    for b_idx in (P_j_bodies + O_j_bodies + M_j_bodies)
                ]
            )
            declaration = CollisionFilterDeclaration().ExcludeBetween(
                geom_set_i, geom_set_j
            )
            manager.Apply(declaration)

    return diagram


def play_motions_on_belief(
    b: state.Belief, U: List[components.CompliantMotion], fname: str = None
):
    meshcat = Meshcat()
    diagram = _make_combined_plant(b, meshcat)
    simulator = Simulator(diagram)
    visualizer = diagram.GetSubsystemByName("meshcat_visualizer(visualizer)")
    visualizer.StartRecording()
    T = 0.0
    for u in U:
        u = ik_solver.update_motion_qd(u)
        T += u.timeout
        for i in range(len(b.particles)):
            controller = diagram.GetSubsystemByName("controller_" + str(i))
            setpoint = diagram.GetSubsystemByName("setpoint_" + str(i))
            setpoint.setpoint = u.q_d
            controller.kp = u.K
        simulator.AdvanceTo(T)
    visualizer.PublishRecording()
    if fname is not None:
        with open(fname, "w") as f:
            f.write(meshcat.StaticHtml())


def show_particle(p: state.Particle):
    u = components.CompliantMotion(
        RigidTransform(), p.X_WG, components.stiff, timeout=0.0
    )
    u = ik_solver.update_motion_qd(u)
    dynamics.simulate(p, u, vis=True)


def show_benchmarks(fname: str):
    import matplotlib.pyplot as plt

    with open(fname, "rb") as f:
        data = pickle.load(f)
    trends = defaultdict(list)
    for params, results in data.items():
        mu, std, sr = utils.mu_std_result(results)
        trends[params[1]].append((params[0], (mu, mu - std, mu + std)))
    for planner, trend in trends.items():
        x_coords = [stat[0] for stat in trend]
        y_coords = [stat[1][0] for stat in trend]
        lb = [stat[1][1] for stat in trend]
        ub = [stat[1][2] for stat in trend]
        plt.fill_between(x_coords, lb, ub, alpha=0.2)
        plt.plot(x_coords, y_coords, label=planner)
    plt.legend()
    plt.savefig(f"{fname[:3]}_plots.png", dpi=1200)
    plt.close()


def playback_result(b, fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    # results = data[("0.009", "True")]
    results = data[("4", "True")]
    for result in results:
        if (result.traj is not None) and len(result.traj) == 4:
            play_motions_on_belief(b, result.traj)
            break
    print("done")
    input()


def visualize_targets(p_nom: state.Particle, targets: List[RigidTransform]):
    for target in targets:
        p_vis = p_nom.deepcopy()
        p_vis.q_r = ik_solver.gripper_to_joint_states(target)
        p_vis._X_WG = None
        u_noop = components.CompliantMotion(
            RigidTransform(), p_vis.X_WG, components.stiff, timeout=0.001
        )
        u_noop.q_d = p_vis.q_r
        # p_vis.env_geom = "assets/floor.sdf"
        # p_vis.X_WO = RigidTransform([0.5, 0, -1.0])
        dynamics.simulate(p_vis, u_noop, vis=True)


def render_graph(nx_graph: nx.Graph, label_dict):
    edge_x = []
    edge_y = []
    pos = nx.spring_layout(nx_graph)
    for edge in nx_graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=0.5, color="#888"), mode="lines"
    )
    node_x = [pos[node][0] for node in nx_graph.nodes()]
    node_y = [pos[node][1] for node in nx_graph.nodes()]
    labels = [label_dict[node] for node in nx_graph.nodes()]
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        marker=dict(
            showscale=False,
            colorscale="YlGnBu",
            reversescale=True,
            color=[],
            size=10,
            line_width=2,
        ),
    )
    node_trace.text = labels
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False)
    fig.write_html("mode_graph.html")


if __name__ == "__main__":
    show_planning_results("pitch_peg_sweep_results.pkl")

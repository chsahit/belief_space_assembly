from typing import List

import numpy as np
from PIL import Image
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    CollisionFilterDeclaration,
    ContactVisualizer,
    DiagramBuilder,
    HPolyhedron,
    Meshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    Simulator,
    VPolytope,
)

import components
import dynamics
import state
from simulation import controller, plant_builder


def _make_combined_plant(b: state.Belief):
    meshcat = Meshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0005)
    parser = Parser(plant)
    parser.package_map().Add("assets", "assets/")
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
    for i, p in enumerate(b.particles):
        P, O, M = instance_list[i]
        plant.SetDefaultPositions(P, p.q_r)
        compliant_controller = builder.AddNamedSystem(
            "controller_" + str(i),
            controller.ControllerSystem(plant, "panda_" + str(i), "block_" + str(i)),
        )
        builder.Connect(
            plant.get_state_output_port(P), compliant_controller.GetInputPort("state")
        )
        builder.Connect(
            compliant_controller.get_output_port(), plant.get_actuation_input_port(P)
        )
    meshcat_vis = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat, MeshcatVisualizerParams()
    )
    # ContactVisualizer.AddToBuilder(builder, plant, meshcat)
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


def play_motions_on_belief(b: state.Belief, U: List[components.CompliantMotion]):
    diagram = _make_combined_plant(b)
    simulator = Simulator(diagram)
    visualizer = diagram.GetSubsystemByName("meshcat_visualizer(visualizer)")
    for i in range(len(b.particles)):
        diagram.GetSubsystemByName("controller_"+str(i)).motion=U[0]
    visualizer.StartRecording()
    simulator.AdvanceTo(U[0].timeout)
    visualizer.PublishRecording()


def _merge_images(images) -> Image:
    base = np.full((1080, 1080, 3), (204, 230, 255), dtype=np.uint8)
    for im in images:
        i = np.array(im)
        bg_color = i[0, 0]
        bg_full = np.full((1080, 1080, 3), bg_color, dtype=np.uint8)
        bg_diff = np.abs(i - bg_full)
        fg_mask = (bg_diff > [1, 1, 1]).any(-1)
        foreground = (i != bg_color).all(-1)
        base[fg_mask] = i[fg_mask]
    return Image.fromarray(base)


def render_motion_set(p_nominal: state.Particle, U: List[components.CompliantMotion]):
    p = p_nominal.deepcopy()
    p.env_geom = "assets/empty_world.sdf"
    U_stiff = [components.CompliantMotion(u.X_GC, u.X_WCd, components.stiff) for u in U]
    P_out = dynamics.f_cspace(p, U_stiff)
    images = [generate_particle_picture(p_i) for p_i in P_out]

    composite = _merge_images(images)
    composite.save("composite.png")


def generate_particle_picture(p: state.Particle, name="test.jpg") -> Image:
    diagram = plant_builder.make_plant_with_cameras(
        p.q_r, p.X_GM, p.X_WO, p.env_geom, p.manip_geom
    )
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.AdvanceTo(0.1)
    logger = diagram.GetSubsystemByName("camera_logger")
    im = Image.fromarray(logger.last_image)
    return im


def show_particle(p: state.Particle):
    diagram, _ = p.make_plant(vis=True)
    simulator = Simulator(diagram)
    meshcat_vis = diagram.GetSubsystemByName("meshcat_visualizer(visualizer)")
    meshcat_vis.StartRecording()
    simulator.AdvanceTo(0.1)
    worst_collision_amt = float("inf")
    wc = None
    for k, v in p.sdf.items():
        if v < worst_collision_amt:
            worst_collision_amt = v
            wc = k
    print(f"{wc=}, {worst_collision_amt=}")
    meshcat_vis.PublishRecording()
    return worst_collision_amt


def plot_motion_sets(sets: List[HPolyhedron]):
    print("constructing list of scatter points")
    if len(sets) > 3:
        raise NotImplementedError(
            f"only supports 3 or fewer sets, called with {len(sets)} sets"
        )
    verts = []
    for mset in sets:
        verts.append(VPolytope(mset).vertices())
    xs = []
    ys = []
    for v_array in verts:
        if v_array.shape[0] != 2:
            raise NotImplementedError(
                "only supports ambient dimension of 2, not {v_array.shape[0]}"
            )
        xs.append([v_array[0, i] for i in range(v_array.shape[1])])
        ys.append([v_array[1, i] for i in range(v_array.shape[1])])
    print("drawing")
    cmap = ["r", "g", "b"]
    import matplotlib.pyplot as plt

    for i in range(2):
        plt.scatter(xs[i], ys[i], c=cmap[i])
    plt.show()


def render_trees(forest: List[components.Tree]):
    import matplotlib.pyplot as plt

    colors = ["r", "g"]
    for i, T in enumerate(forest):
        for v in T.nodes:
            if v.u_pred is None:
                continue
            xs = (v.u_pred.u.X_WCd.translation()[0], v.u.X_WCd.translation()[0])
            ys = (v.u_pred.u.X_WCd.translation()[2], v.u.X_WCd.translation()[2])
            plt.plot(xs, ys, marker="o", c=colors[i])
    print("not showing...")
    # plt.show()

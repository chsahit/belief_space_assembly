import pickle
from typing import List

import numpy as np
from PIL import Image
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    CollisionFilterDeclaration,
    ContactModel,
    ContactVisualizer,
    DiagramBuilder,
    HPolyhedron,
    IllustrationProperties,
    Meshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    RigidTransform,
    Role,
    RoleAssign,
    Simulator,
    VPolytope,
)

import components
import dynamics
import state
import utils
from simulation import controller, ik_solver, plant_builder


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
    diagram, _ = p.make_plant(vis=True)
    simulator = Simulator(diagram)
    simulator.Initialize()
    meshcat_vis = diagram.GetSubsystemByName("meshcat_visualizer(visualizer)")
    meshcat_vis.StartRecording()
    simulator.AdvanceTo(0.001)
    meshcat_vis.PublishRecording()


def show_planning_results(fname: str):
    print("rendering results")
    import matplotlib.pyplot as plt

    with open(fname, "rb") as f:
        data = pickle.load(f)

    utils.envelope_analysis(data)
    line_compliant_x, line_compliant_y = [], []
    line_stiff_x, line_stiff_y = [], []
    compliant_std_low, compliant_std_high = [], []
    stiff_std_low, stiff_std_high = [], []
    line_ngp_x, line_ngp_y = [], []
    ngp_std_low, ngp_std_high = [], []
    for params, results in data.items():
        deviation = 2 * float(params[0])
        # results_succ = [result in results if (result.traj is not None)]
        # if len(results_succ) == 0:
        #     continue
        mu, std = utils.mu_std_result(results)
        if params[1] == "True" and params[2] == "True":
            line_compliant_x.append(deviation)
            line_compliant_y.append(mu)
            compliant_std_low.append(mu - std)
            compliant_std_high.append(mu + std)
        elif params[1] == "True" and params[2] == "False":
            line_ngp_x.append(deviation)
            line_ngp_y.append(mu)
            ngp_std_low.append(mu - std)
            ngp_std_high.append(mu + std)
        else:
            line_stiff_x.append(deviation)
            line_stiff_y.append(mu)
            stiff_std_low.append(mu - std)
            stiff_std_high.append(mu + std)

    plt.fill_between(
        line_compliant_x, compliant_std_low, compliant_std_high, alpha=0.2, color="b"
    )
    plt.plot(line_compliant_x, line_compliant_y, color="b", label="ours")

    plt.fill_between(line_stiff_x, stiff_std_low, stiff_std_high, alpha=0.2, color="g")
    plt.plot(line_stiff_x, line_stiff_y, color="g", label="no stiffness")
    plt.title("The effect of Uncertainty on Planning")
    plt.xlabel("Uncertainty (degrees)")
    plt.ylabel("Planning Time")
    plt.legend()
    plt.show()
    print("done")


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
        p_vis.env_geom = "assets/floor.sdf"
        p_vis.X_WO = RigidTransform([0.5, 0, -1.0])
        dynamics.simulate(p_vis, u_noop, vis=True)


if __name__ == "__main__":
    show_planning_results("pitch_peg_sweep_results.pkl")

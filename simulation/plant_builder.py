from typing import Dict

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    ContactModel,
    CoulombFriction,
    DiagramBuilder,
    DiscreteContactSolver,
    Meshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    MultibodyPlant,
    Parser,
    ProximityProperties,
    RigidTransform,
    Sphere,
    System,
)

import utils
from simulation import controller

timestep = 0.005
contact_model = ContactModel.kPoint  # ContactModel.kHydroelasticWithFallback


def generate_collision_spheres() -> Dict[str, RigidTransform]:
    epsilon = 1e-5
    id_to_rt = dict()
    for i, x in enumerate([-0.03, 0.03]):
        for j, y in enumerate([-0.03, 0.03]):
            for k, z in enumerate([-0.075, 0.075, 0.05]):
                rt = RigidTransform([x, y, z])
                name = str(i) + str(j) + str(k)
                id_to_rt["block::" + name] = rt
    return id_to_rt


def weld_geometries(plant: MultibodyPlant, X_GB: RigidTransform, X_WO: RigidTransform):
    plant.WeldFrames(
        frame_on_parent_F=plant.world_frame(),
        frame_on_child_M=plant.GetFrameByName("panda_link0"),
        X_FM=utils.xyz_rpy_deg([0, 0, 0], [0, 0, 0]),
    )
    plant.WeldFrames(
        frame_on_parent_F=plant.world_frame(),
        frame_on_child_M=plant.GetFrameByName("bin_base"),
        X_FM=X_WO,
    )
    plant.WeldFrames(
        frame_on_parent_F=plant.GetFrameByName("panda_hand"),
        frame_on_child_M=plant.GetFrameByName("base_link"),
        X_FM=X_GB,
    )


def make_plant(
    q_r: np.ndarray,
    X_GB: RigidTransform,
    X_WO: RigidTransform,
    env_geom: str,
    manip_geom: str,
    collision_check: bool = False,
    vis: bool = False,
) -> System:

    # Plant hyperparameters
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, timestep)
    plant.set_discrete_contact_solver(DiscreteContactSolver.kSap)
    plant.set_contact_model(contact_model)
    plant.set_penetration_allowance(0.0005)

    # load and add rigidbodies to plant
    parser = Parser(plant)
    parser.package_map().Add("assets", "assets/")
    panda = parser.AddModelFromFile("assets/panda_arm_hand.urdf", model_name="panda")
    env_geometry = parser.AddAllModelsFromFile(env_geom)[0]
    manipuland = parser.AddModelFromFile(manip_geom, model_name="block")
    if collision_check:
        sphere_map = generate_collision_spheres()
        manipuland_body = plant.get_body(plant.GetBodyIndices(manipuland)[0])
        for (name, rt) in sphere_map.items():
            # props = ProximityProperties()
            # props.AddProperty("material", "coulomb_friction", CoulombFriction(0.0, 0.0))
            plant.RegisterCollisionGeometry(
                manipuland_body, rt, Sphere(1e-5), name[-3:], CoulombFriction(0.0, 0.0)
            )
    weld_geometries(plant, X_GB, X_WO)
    plant.Finalize()
    plant.SetDefaultPositions(panda, q_r)

    # connect controller
    compliant_controller = builder.AddNamedSystem(
        "controller", controller.ControllerSystem(plant)
    )
    builder.Connect(
        plant.get_state_output_port(panda), compliant_controller.GetInputPort("state")
    )
    builder.Connect(
        compliant_controller.get_output_port(), plant.get_actuation_input_port(panda)
    )
    builder.Connect(
        scene_graph.get_query_output_port(),
        compliant_controller.GetInputPort("geom_query"),
    )
    if vis:
        meshcat = Meshcat()
        meshcat_vis = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, meshcat, MeshcatVisualizerParams()
        )
    diagram = builder.Build()
    return diagram

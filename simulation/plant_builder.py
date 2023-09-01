import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    ContactModel,
    DiagramBuilder,
    DiscreteContactSolver,
    Meshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    MultibodyPlant,
    Parser,
    RigidTransform,
    System,
)

import utils
from simulation import controller

timestep = 0.005
contact_model = ContactModel.kPoint  # ContactModel.kHydroelasticWithFallback


def add_collision_spheres_to_peg():
    pass


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
    collision_spheres: bool = False,
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
    if collision_spheres:
        add_collision_spheres_to_peg()
    weld_geometries(plant, X_GB, X_WO)
    plant.Finalize()
    plant.SetDefaultPositions(panda, q_r)

    # connect controller
    compliant_controller = builder.AddSystem(controller.ControllerSystem(plant))
    builder.Connect(
        plant.get_state_output_port(panda), compliant_controller.GetInputPort("state")
    )
    builder.Connect(
        compliant_controller.get_output_port(), plant.get_actuation_input_port(panda)
    )
    if vis:
        meshcat = Meshcat()
        meshcat_vis = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, meshcat, MeshcatVisualizerParams()
        )
    diagram = builder.Build()
    return diagram

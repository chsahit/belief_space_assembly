import os
from typing import Dict, List, Tuple

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    CameraInfo,
    ClippingRange,
    ContactModel,
    ContactVisualizer,
    CoulombFriction,
    DepthRange,
    DepthRenderCamera,
    Diagram,
    DiagramBuilder,
    DiscreteContactApproximation,
    FirstOrderLowPassFilter,
    JointActuatorIndex,
    JointStiffnessController,
    MakeRenderEngineGl,
    MakeRenderEngineVtk,
    Meshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    ModelInstanceIndex,
    MultibodyPlant,
    Parser,
    PdControllerGains,
    ProximityProperties,
    RenderCameraCore,
    RenderEngineGlParams,
    RenderEngineVtkParams,
    RgbdSensor,
    RgbdSensorDiscrete,
    RigidTransform,
    Role,
    RoleAssign,
    SceneGraph,
    Sphere,
)

import utils
from simulation import annotate_geoms, controller, geometry_monitor, image_logger
from simulation import joint_impedance_controller as jc
from simulation import playback_controller

timestep = 0.005
contact_model = ContactModel.kHydroelasticWithFallback
contact_approx = DiscreteContactApproximation.kSimilar


def init_plant(
    builder,
    timestep=0.005,
    contact_model=ContactModel.kHydroelasticWithFallback,
    contact_approx=DiscreteContactApproximation.kLagged,
):
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, timestep)
    plant.set_contact_model(contact_model)
    if timestep > 0:
        # plant.set_penetration_allowance(0.0005)
        plant.set_discrete_contact_approximation(contact_approx)
    parser = Parser(plant)
    parser.package_map().Add("assets", "assets/")
    return plant, scene_graph, parser


def wire_controller(
    is_cartesian: bool,
    panda: ModelInstanceIndex,
    controller_name: str,
    panda_name: str,
    block_name: str,
    builder,
    plant,
):
    compliant_controller = None
    if is_cartesian:
        compliant_controller = builder.AddNamedSystem(
            controller_name,
            playback_controller.PlaybackController(plant, panda_name),
        )
        builder.Connect(
            compliant_controller.get_output_port(),
            plant.get_actuation_input_port(panda),
        )
    else:
        compliant_controller = builder.AddNamedSystem(
            controller_name, jc.JointImpedanceController(plant, panda_name)
        )
        builder.Connect(
            compliant_controller.GetOutputPort("q_d"),
            plant.get_desired_state_input_port(panda),
        )
        builder.Connect(
            compliant_controller.GetOutputPort("gravity_ff"),
            plant.get_actuation_input_port(panda),
        )

    builder.Connect(
        plant.get_state_output_port(panda),
        compliant_controller.GetInputPort("state"),
    )
    return compliant_controller


def _drop_reflected_inertia(plant, panda):
    ja_indices = plant.GetJointActuatorIndices(panda)
    for ja_idx in ja_indices:
        ja = plant.get_joint_actuator(ja_idx)
        ja.set_default_rotor_inertia(0.0)
        ja.set_default_gear_ratio(0.0)


def _weld_geometries(
    plant: MultibodyPlant,
    X_GM: RigidTransform,
    X_WO: RigidTransform,
    panda_instance: ModelInstanceIndex,
    block_instance: ModelInstanceIndex,
    obj_instance: ModelInstanceIndex,
):
    plant.WeldFrames(
        frame_on_parent_F=plant.world_frame(),
        frame_on_child_M=plant.GetFrameByName("panda_link0", panda_instance),
        X_FM=utils.xyz_rpy_deg([0, 0, 0], [0, 0, 0]),
    )
    plant.WeldFrames(
        frame_on_parent_F=plant.world_frame(),
        frame_on_child_M=plant.GetFrameByName("bin_base", obj_instance),
        X_FM=X_WO,
    )
    plant.WeldFrames(
        frame_on_parent_F=plant.GetFrameByName("panda_hand", panda_instance),
        frame_on_child_M=plant.GetFrameByName("base_link", block_instance),
        X_FM=X_GM,
    )


def make_plant(
    q_r: np.ndarray,
    X_GM: RigidTransform,
    X_WO: RigidTransform,
    env_geom: str,
    manip_geom: str,
    collision_check: bool = False,
    vis: bool = False,
    mu: float = 0.0,
    meshcat_instance=None,
    gains=None,
) -> Tuple[Diagram, Meshcat]:
    builder, _, _, meshcat = _construct_diagram(
        q_r,
        X_GM,
        X_WO,
        env_geom,
        manip_geom,
        collision_check=collision_check,
        vis=vis,
        mu=mu,
        meshcat_instance=meshcat_instance,
        gains=gains,
    )
    diagram = builder.Build()
    return diagram, meshcat


def _set_frictions(
    plant: MultibodyPlant,
    scene_graph: SceneGraph,
    model_instances: List[ModelInstanceIndex],
    mu_d: float,
):
    inspector = (
        scene_graph.get_query_output_port()
        .Eval(scene_graph.CreateDefaultContext())
        .inspector()
    )
    for model in model_instances:
        for body_id in plant.GetBodyIndices(model):
            frame_id = plant.GetBodyFrameIdOrThrow(body_id)
            geometry_ids = inspector.GetGeometries(frame_id, Role.kProximity)
            for g_id in geometry_ids:
                prop = inspector.GetProximityProperties(g_id)
                new_props = ProximityProperties(prop)
                friction_property = CoulombFriction(mu_d, mu_d)
                new_props.UpdateProperty(
                    "material", "coulomb_friction", friction_property
                )
                scene_graph.AssignRole(
                    plant.get_source_id(), g_id, new_props, RoleAssign.kReplace
                )


def _construct_diagram(
    q_r: np.ndarray,
    X_GM: RigidTransform,
    X_WO: RigidTransform,
    env_geom: str,
    manip_geom: str,
    collision_check: bool = False,
    vis: bool = False,
    mu: float = 0.0,
    meshcat_instance=None,
    gains=None,
) -> Tuple[DiagramBuilder, MultibodyPlant, SceneGraph, Meshcat]:
    builder = DiagramBuilder()
    plant, scene_graph, parser = init_plant(builder)

    # load and add rigidbodies to plant
    panda = parser.AddModels("assets/panda_arm_hand.urdf")[0]
    panda_name = "panda"
    plant.RenameModelInstance(panda, panda_name)
    env_geometry = parser.AddModels(env_geom)[0]
    # manipuland = parser.AddModelFromFile(manip_geom, model_name="block")
    manipuland = parser.AddModels(manip_geom)[0]
    plant.RenameModelInstance(manipuland, "block")
    _weld_geometries(plant, X_GM, X_WO, panda, manipuland, env_geometry)
    # _mu = mu if gains is not None else mu - 0.01
    _set_frictions(plant, scene_graph, [env_geometry, manipuland], mu)
    for i, ja_index in enumerate(list(range(7))):
        ja = plant.get_joint_actuator(JointActuatorIndex(ja_index))
        if gains is not None:
            ja.set_controller_gains(
                PdControllerGains(p=gains[i, i], d=10 * np.sqrt(gains[i, i]))
            )
    plant.Finalize()
    _drop_reflected_inertia(plant, panda)
    plant.SetDefaultPositions(panda, q_r)

    if collision_check:
        geom_monitor = builder.AddNamedSystem(
            "geom_monitor", geometry_monitor.GeometryMonitor(plant)
        )
        builder.Connect(
            scene_graph.get_query_output_port(),
            geom_monitor.GetInputPort("geom_query"),
        )
        builder.Connect(
            plant.get_state_output_port(panda),
            geom_monitor.GetInputPort("state"),
        )

    # connect controller
    is_cartesian = gains is None
    wire_controller(is_cartesian, panda, "controller", "panda", "block", builder, plant)
    meshcat = meshcat_instance
    if vis:
        if meshcat is None:
            meshcat = Meshcat()
        meshcat_vis = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, meshcat, MeshcatVisualizerParams()
        )
        ContactVisualizer.AddToBuilder(builder, plant, meshcat)
    return builder, plant, scene_graph, meshcat

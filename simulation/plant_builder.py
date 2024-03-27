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
from simulation import controller, full_joint_stiffness, geometry_monitor

timestep = 0.005
contact_model = ContactModel.kHydroelasticWithFallback


def init_plant(
    builder,
    timestep=0.005,
    contact_model=ContactModel.kHydroelasticWithFallback,
    contact_approx=DiscreteContactApproximation.kLagged,
):
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, timestep)
    plant.set_contact_model(contact_model)
    plant.set_penetration_allowance(0.0005)
    if timestep > 0:
        plant.set_discrete_contact_approximation(contact_approx)
    parser = Parser(plant)
    parser.package_map().Add("assets", "assets/")
    return plant, scene_graph, parser


def wire_controller(
    panda: ModelInstanceIndex,
    controller_name: str,
    panda_name: str,
    setpoint_name: str,
    builder,
    plant,
):
    compliant_controller = builder.AddNamedSystem(
        controller_name,
        full_joint_stiffness.JointStiffnessController(
            plant, None, panda_name=panda_name
        ),
    )
    fixblock = builder.AddNamedSystem(
        setpoint_name, full_joint_stiffness.FixedVal(None)
    )
    builder.Connect(
        plant.get_state_output_port(panda),
        compliant_controller.get_input_port_estimated_state(),
    )
    builder.Connect(
        fixblock.get_output_port(),
        compliant_controller.get_input_port_desired_state(),
    )
    builder.Connect(
        compliant_controller.get_output_port_generalized_force(),
        plant.get_actuation_input_port(panda),
    )

    return compliant_controller


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
) -> Tuple[DiagramBuilder, MultibodyPlant, SceneGraph, Meshcat]:
    builder = DiagramBuilder()
    plant, scene_graph, parser = init_plant(builder)

    # load and add rigidbodies to plant
    panda = parser.AddModels("assets/panda_arm_hand.urdf")[0]
    panda_name = "panda"
    plant.RenameModelInstance(panda, panda_name)
    env_geometry = parser.AddModels(env_geom)[0]
    manipuland = parser.AddModels(manip_geom)[0]
    plant.RenameModelInstance(manipuland, "block")
    _weld_geometries(plant, X_GM, X_WO, panda, manipuland, env_geometry)
    _set_frictions(plant, scene_graph, [env_geometry, manipuland], mu)
    plant.Finalize()
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
    wire_controller(panda, "controller", "panda", "setpoint", builder, plant)
    meshcat = meshcat_instance
    if vis:
        if meshcat is None:
            meshcat = Meshcat()
        meshcat_vis = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, meshcat, MeshcatVisualizerParams()
        )
        ContactVisualizer.AddToBuilder(builder, plant, meshcat)
    return builder, plant, scene_graph, meshcat

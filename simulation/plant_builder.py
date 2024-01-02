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

timestep = 0
timestep = 0.0002
contact_model = ContactModel.kPoint  # ContactModel.kHydroelasticWithFallback


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


def make_plant_with_cameras(
    q_r: np.ndarray,
    X_GM: RigidTransform,
    X_WO: RigidTransform,
    env_geom: str,
    manip_geom: str,
    vis: bool = False,
) -> Diagram:
    builder, plant, scene_graph, meshcat = _construct_diagram(
        q_r, X_GM, X_WO, env_geom, manip_geom, vis=vis
    )
    from pyvirtualdisplay import Display

    vd = Display(visible=0, size=(1400, 900))
    vd.start()
    scene_graph.AddRenderer("renderer", MakeRenderEngineVtk(RenderEngineVtkParams()))
    depth_cam = DepthRenderCamera(
        RenderCameraCore(
            "renderer",
            CameraInfo(width=1080, height=1080, fov_y=np.pi / 4),
            ClippingRange(0.01, 10.0),
            RigidTransform(),
        ),
        DepthRange(0.01, 10.0),
    )
    X_PB = utils.xyz_rpy_deg([2.0, 0, 0.1], [-90, 0, 90])
    # X_PB = RigidTransform([0, 0, 0.15])
    world_idx = plant.GetBodyFrameIdOrThrow(plant.world_body().index())
    loc = plant.GetBodyFrameIdOrThrow(plant.GetBodyByName("panda_hand").index())
    sensor = RgbdSensor(
        world_idx,
        X_PB,
        depth_camera=depth_cam,
        show_window=False,
    )
    discrete_sensor = RgbdSensorDiscrete(sensor, 0.1, False)
    discrete_sensor = builder.AddSystem(discrete_sensor)
    builder.Connect(
        scene_graph.get_query_output_port(), discrete_sensor.query_object_input_port()
    )
    cam_logger = builder.AddNamedSystem("camera_logger", image_logger.ImageLogger())

    builder.Connect(
        discrete_sensor.color_image_output_port(), cam_logger.GetInputPort("rbg_in")
    )
    diagram = builder.Build()
    return diagram


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

    print("building")
    # Plant hyperparameters
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, timestep)
    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)
    plant.set_contact_model(contact_model)
    plant.set_penetration_allowance(0.0005)

    # load and add rigidbodies to plant
    parser = Parser(plant)
    parser.package_map().Add("assets", "assets/")
    panda = parser.AddModels("assets/panda_arm_hand.urdf")[0]
    panda_name = "panda"
    plant.RenameModelInstance(panda, panda_name)
    env_geometry = parser.AddModels(env_geom)[0]
    # manipuland = parser.AddModelFromFile(manip_geom, model_name="block")
    manipuland = parser.AddModels(manip_geom)[0]
    plant.RenameModelInstance(manipuland, "block")
    if collision_check:
        sphere_map = annotate_geoms.annotate(manip_geom)
        manipuland_body = plant.get_body(plant.GetBodyIndices(manipuland)[0])
        for (name, rt) in sphere_map.items():
            continue
            plant.RegisterCollisionGeometry(
                manipuland_body, rt, Sphere(1e-5), name[-3:], CoulombFriction(0.0, 0.0)
            )
    _weld_geometries(plant, X_GM, X_WO, panda, manipuland, env_geometry)
    _set_frictions(plant, scene_graph, [env_geometry, manipuland], mu)
    for i, ja_index in enumerate(
        list(range(7))
    ):  # enumerate(plant.GetJointActuatorIndices(panda)):
        ja = plant.get_joint_actuator(JointActuatorIndex(ja_index))
        # TODO: jacobian...
        if gains is not None:
            ja.set_controller_gains(
                PdControllerGains(p=gains[i, i], d=4 * np.sqrt(gains[i, i]))
            )
    # assert False
    # finger_l = plant.GetJointByName("")
    # finger_r = plant.GetJointByName("")
    plant.Finalize()
    ja_indices = plant.GetJointActuatorIndices(panda)
    for ja_idx in ja_indices:
        ja = plant.get_joint_actuator(ja_idx)
        ja.set_default_rotor_inertia(0.0)
        ja.set_default_gear_ratio(0.0)

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
    out_size = 7
    if gains is not None:
        out_size = 14
    compliant_controller = builder.AddNamedSystem(
        "controller",
        controller.ControllerSystem(plant, panda_name, "block", out_size=out_size),
    )
    builder.Connect(
        plant.get_state_output_port(panda),
        compliant_controller.GetInputPort("state"),
    )

    if gains is None:
        lowpass = builder.AddSystem(FirstOrderLowPassFilter(0.005, size=7))
        builder.Connect(
            compliant_controller.get_output_port(), lowpass.get_input_port()
        )
        builder.Connect(
            lowpass.get_output_port(), plant.get_actuation_input_port(panda)
        )
    else:
        builder.Connect(
            compliant_controller.get_output_port(),
            plant.get_desired_state_input_port(panda),
        )
    meshcat = meshcat_instance
    if vis:
        if meshcat is None:
            meshcat = Meshcat()
        meshcat_vis = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, meshcat, MeshcatVisualizerParams()
        )
        ContactVisualizer.AddToBuilder(builder, plant, meshcat)
    return builder, plant, scene_graph, meshcat

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
    DiscreteContactSolver,
    InverseDynamics,
    MakeRenderEngineGl,
    MakeRenderEngineVtk,
    Meshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    ModelInstanceIndex,
    MultibodyPlant,
    Parser,
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
    ZeroOrderHold,
)

import utils
from simulation import controller, geometry_monitor, image_logger

timestep = 0.001
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


def _weld_geometries(plant: MultibodyPlant, X_GM: RigidTransform, X_WO: RigidTransform):
    plant.WeldFrames(
        frame_on_parent_F=plant.world_frame(),
        frame_on_child_M=plant.GetFrameByName("panda_link0"),
        X_FM=utils.xyz_rpy_deg([0, 0, 0], [0, 0, 0]),
    )
    """
    plant.WeldFrames(
        frame_on_parent_F=plant.world_frame(),
        frame_on_child_M=plant.GetFrameByName("bin_base"),
        X_FM=X_WO,
    )
    plant.WeldFrames(
        frame_on_parent_F=plant.GetFrameByName("panda_hand"),
        frame_on_child_M=plant.GetFrameByName("base_link"),
        X_FM=X_GM,
    )
    """


def make_plant(
    q_r: np.ndarray,
    X_GM: RigidTransform,
    X_WO: RigidTransform,
    env_geom: str,
    manip_geom: str,
    collision_check: bool = False,
    vis: bool = False,
    mu: float = 0.0,
) -> Diagram:
    builder, _, _, meshcat = _construct_diagram(
        q_r,
        X_GM,
        X_WO,
        env_geom,
        manip_geom,
        collision_check=collision_check,
        vis=vis,
        mu=mu,
    )
    diagram = builder.Build()
    if vis:
        return diagram, meshcat
    return diagram


def make_plant_with_cameras(
    q_r: np.ndarray,
    X_GM: RigidTransform,
    X_WO: RigidTransform,
    env_geom: str,
    manip_geom: str,
    vis: bool = False,
) -> Diagram:
    builder, plant, scene_graph, _ = _construct_diagram(
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
) -> Tuple[DiagramBuilder, MultibodyPlant, SceneGraph, Meshcat]:

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
    """
    env_geometry = parser.AddAllModelsFromFile(env_geom)[0]
    manipuland = parser.AddModelFromFile(manip_geom, model_name="block")
    if collision_check:
        sphere_map = generate_collision_spheres()
        manipuland_body = plant.get_body(plant.GetBodyIndices(manipuland)[0])
        for (name, rt) in sphere_map.items():
            plant.RegisterCollisionGeometry(
                manipuland_body, rt, Sphere(1e-5), name[-3:], CoulombFriction(0.0, 0.0)
            )
    _set_frictions(plant, scene_graph, [env_geometry, manipuland], mu)
    """
    _weld_geometries(plant, X_GM, X_WO)
    plant.Finalize()
    plant.SetDefaultPositions(panda, q_r)

    if collision_check:
        geom_monitor = builder.AddNamedSystem(
            "geom_monitor", geometry_monitor.GeometryMonitor()
        )
        builder.Connect(
            scene_graph.get_query_output_port(),
            geom_monitor.GetInputPort("geom_query"),
        )

    # connect controller
    compliant_controller = builder.AddNamedSystem(
        "controller", controller.VirtualSpringDamper(plant)
    )
    inv_dynamics = builder.AddSystem(InverseDynamics(plant))
    hold = builder.AddSystem(ZeroOrderHold(0.001, 9))
    builder.Connect(
        plant.get_state_output_port(panda), compliant_controller.GetInputPort("state")
    )
    builder.Connect(
        plant.get_generalized_acceleration_output_port(),
        hold.get_input_port()
    )
    builder.Connect(
        hold.get_output_port(),
        compliant_controller.GetInputPort("acceleration")
    )
    builder.Connect(
        compliant_controller.get_output_port(), inv_dynamics.get_input_port_desired_acceleration()# plant.get_actuation_input_port(panda)
    )
    builder.Connect(
        plant.get_state_output_port(), inv_dynamics.get_input_port_estimated_state()
    )
    builder.Connect(
        inv_dynamics.get_output_port_force(), plant.get_actuation_input_port(panda)
    )
    meshcat = None
    if vis:
        meshcat = Meshcat()
        meshcat_vis = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, meshcat, MeshcatVisualizerParams()
        )
        ContactVisualizer.AddToBuilder(builder, plant, meshcat)
    return builder, plant, scene_graph, meshcat

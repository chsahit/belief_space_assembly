from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Diagram,
    DiagramBuilder,
    GeometryId,
    InverseKinematics,
    MultibodyPlant,
    Parser,
    RigidTransform,
    Role,
    RollPitchYaw,
    RotationMatrix,
    Simulator,
    Solve,
)

import components
import utils
# import visualize
from simulation import annotate_geoms


def gripper_to_joint_states(
    X_WG: RigidTransform, plant: MultibodyPlant = None
) -> np.ndarray:
    if plant is None:
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.1)
        parser = Parser(plant)
        parser.package_map().Add("assets", "assets/")
        panda = parser.AddModels("assets/panda_arm_hand.urdf")
        plant.WeldFrames(
            frame_on_parent_F=plant.world_frame(),
            frame_on_child_M=plant.GetFrameByName("panda_link0"),
            X_FM=utils.xyz_rpy_deg([0, 0, 0], [0, 0, 0]),
        )
        plant.Finalize()

    ik = InverseKinematics(plant)
    ik.AddPositionConstraint(
        plant.GetFrameByName("panda_hand"),
        [0, 0, 0],
        plant.world_frame(),
        X_WG.translation(),
        X_WG.translation(),
    )
    ik.AddOrientationConstraint(
        plant.GetFrameByName("panda_hand"),
        RotationMatrix(),
        plant.world_frame(),
        X_WG.rotation(),
        0.0,
    )
    prog = ik.get_mutable_prog()
    q = ik.q()
    q0_guess = [
        0.0796904,
        0.18628879,
        -0.07548908,
        -2.42085905,
        0.06961755,
        2.52396334,
        0.6796144,
        0.04287501,
        0.04266755,
    ]
    prog.AddQuadraticErrorCost(np.identity(len(q)), q0_guess, q)
    prog.SetInitialGuess(q, q0_guess)
    result = Solve(ik.prog())
    soln = result.GetSolution(q)
    return soln


def get_geometry_ids(diagram: Diagram) -> Tuple[GeometryId, Dict[str, GeometryId]]:
    plant = diagram.GetSubsystemByName("plant")
    scene_graph = diagram.GetSubsystemByName("scene_graph")
    inspector = (
        scene_graph.get_query_output_port()
        .Eval(scene_graph.CreateDefaultContext())
        .inspector()
    )
    manipuland_body_frame = plant.GetBodyFrameIdOrThrow(
        plant.GetBodyByName("base_link").index()
    )
    manipuland_id = inspector.GetGeometries(manipuland_body_frame, Role.kProximity)[0]
    env_body_frame = plant.GetBodyFrameIdOrThrow(
        plant.GetBodyByName("bin_base").index()
    )
    env_g_ids = inspector.GetGeometries(env_body_frame, Role.kProximity)
    g_id_map = dict()
    for g_id in env_g_ids:
        name = inspector.GetName(g_id)
        g_id_map[name] = g_id
    return manipuland_id, g_id_map


def project_manipuland_to_contacts(
    p: state.Particle,
    CF_d: components.ContactState,
    fallback: bool = False,
    vis: bool = False,
) -> RigidTransform:
    p_aligned = axis_align_particle(p)
    diagram, _ = p_aligned.make_plant()
    plant = diagram.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(diagram.CreateDefaultContext())
    constraints = p.constraints
    ik = InverseKinematics(plant, plant_context)
    corner_map = annotate_geoms.annotate(p.manip_geom)

    W = plant.world_frame()
    M = plant.GetBodyByName("base_link").body_frame()
    G = plant.GetBodyByName("panda_hand").body_frame()
    for env_poly, object_corner in CF_d:
        p_MP = corner_map[object_corner].translation()
        A, b = constraints[env_poly]
        ik.AddPolyhedronConstraint(W, M, p_MP, A, b)

    q = ik.q()
    prog = ik.get_mutable_prog()
    prog.SetInitialGuess(q, p_aligned.q_r)

    p_WM = p.X_WG.multiply(p_aligned.X_GM).translation()
    ik.AddPositionCost(W, p_WM, M, np.zeros((3,)), np.identity(3))
    # R_WM = p.X_WG.multiply(p.X_GM).rotation()
    # ik.AddOrientationCost(W, R_WM, M, RotationMatrix(np.eye(3)), 0.01)

    g_manipuland, g_ids = get_geometry_ids(diagram)
    if fallback:
        ap = -1e-5
    else:
        ap = -1e-10
    for k in g_ids.keys():
        if k not in str(CF_d):
            ik.AddDistanceConstraint((g_ids[k], g_manipuland), ap, np.inf)
        else:
            ik.AddDistanceConstraint((g_ids[k], g_manipuland), -1e-4, np.inf)
    try:
        result = Solve(ik.prog())
        X_WG_out = plant.CalcRelativeTransform(plant_context, W, G)
        if not result.is_success():
            if not fallback:
                p_aligned = axis_align_particle(p)
                return project_manipuland_to_contacts(p_aligned, CF_d, fallback=True)
            else:
                print(f"warning, ik solve failed. {X_WG_out}")
    except Exception as e:
        print("IK SOLVER EXCEPTION")
        if not fallback:
            p_aligned = axis_align_particle(p)
            return project_manipuland_to_contacts(p_aligned, CF_d, fallback=True)
        else:
            print("info on p_aligned: \n", utils.rt_to_str(p_aligned.X_WG))
            print(f"warning, ik solve failed.")
            return None
    p_aligned.q_r = plant.GetPositions(
        plant_context, plant.GetModelInstanceByName("panda")
    )
    X_WG_out = step_sim(p_aligned)
    if vis:
        print("showing ik result")
        print(utils.rt_to_str(X_WG_out))
        visualize.show_particle(p_aligned)
        # p_aligned.env_geom = "assets/empty_world.sdf"
    return X_WG_out


def axis_align_particle(p: state.Particle) -> state.Particle:
    X_WM = p.X_WG.multiply(p.X_GM)
    X_WM_aa = RigidTransform(X_WM.GetAsMatrix4())
    # TODO: make a manip_geom dataclass so i dont have these if/elifs everywhere
    if p.manip_geom == "assets/peg.urdf":
        nominal_manipuland_orientation = RollPitchYaw(np.array([np.pi, 0, 0]))
    elif p.manip_geom == "assets/moving_puzzle.sdf":
        nominal_manipuland_orientation = RollPitchYaw(np.array([0, 0, 0]))
    else:
        raise NotImplementedError("bad geometry instance")
    X_WM_aa.set_rotation(nominal_manipuland_orientation)
    X_WM_aa_translation = X_WM_aa.translation().copy()
    X_WM_aa_translation[0] = 0.5
    X_WM_aa_translation[2] = 0.15
    X_WM_aa.set_translation(X_WM_aa_translation)
    X_WG_aa = X_WM_aa.multiply(p.X_GM.inverse())
    q_r_aa = gripper_to_joint_states(X_WG_aa)
    new_p = p.deepcopy()
    new_p.q_r = q_r_aa
    return new_p


def step_sim(p: state.Particle) -> RigidTransform:
    # print("before step: ")
    # visualize.show_particle(p)
    diagram, _ = p.make_plant()
    plant = diagram.GetSubsystemByName("plant")
    simulator = Simulator(diagram)
    plant_context = plant.GetMyContextFromRoot(simulator.get_mutable_context())
    simulator.Initialize()
    simulator.AdvanceTo(0.2)
    X_WG = plant.CalcRelativeTransform(
        plant_context,
        plant.world_frame(),
        plant.GetBodyByName("panda_hand").body_frame(),
    )
    q_r_T = plant.GetPositions(plant_context, plant.GetModelInstanceByName("panda"))
    p_next = p.deepcopy()
    p_next.q_r = q_r_T
    # print("after step: ")
    # visualize.show_particle(p_next)
    return X_WG

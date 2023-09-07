from __future__ import annotations

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    InverseKinematics,
    MultibodyPlant,
    Parser,
    RigidTransform,
    RotationMatrix,
    Solve,
)

import components
import utils
from simulation import plant_builder


def gripper_to_joint_states(
    X_WG: RigidTransform, plant: MultibodyPlant = None
) -> np.ndarray:
    if plant is None:
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.1)
        parser = Parser(plant)
        parser.package_map().Add("assets", "assets/")
        panda = parser.AddModelFromFile(
            "assets/panda_arm_hand.urdf", model_name="panda"
        )
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


def project_manipuland_to_contacts(
    p: belief_state.Particle, CF_d: components.ContactState
) -> RigidTransform:
    diagram = p.make_plant()
    plant = diagram.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(diagram.CreateDefaultContext())
    constraints = p.constraints
    ik = InverseKinematics(plant, plant_context)
    corner_map = plant_builder.generate_collision_spheres()

    W = plant.world_frame()
    M = plant.GetBodyByName("base_link").body_frame()
    G = plant.GetBodyByName("panda_hand").body_frame()
    for env_poly, object_corner in CF_d:
        p_MP = corner_map[object_corner].translation()
        A, b = constraints[env_poly]
        ik.AddPolyhedronConstraint(W, M, p_MP, A, b)

    q = ik.q()
    prog = ik.get_mutable_prog()
    prog.SetInitialGuess(q, p.q_r)

    p_WM = [0.5, 0.0, 0.225]
    ik.AddPositionCost(W, p_WM, M, np.zeros((3,)), np.identity(3))

    try:
        result = Solve(ik.prog())
        if not result.is_success():
            print("warning, ik solve failed")
    except:
        return None

    return plant.CalcRelativeTransform(plant_context, W, G)

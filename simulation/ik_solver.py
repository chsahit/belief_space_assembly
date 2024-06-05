from __future__ import annotations

from typing import List

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    InverseKinematics,
    MultibodyPlant,
    Parser,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    Solve,
)


def xyz_rpy_deg(xyz: np.ndarray, rpy_deg: List[float]) -> RigidTransform:
    """Shorthand for defining a pose."""
    rpy_deg_np = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg_np * np.pi / 180), xyz)


# puzzle: finger_width=0.015
# peg: finger_width=0.03
def gripper_to_joint_states(X_WG: RigidTransform, finger_width=0.03) -> np.ndarray:
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.1)
    parser = Parser(plant)
    parser.package_map().Add("assets", "assets/")
    parser.AddModels("assets/panda_arm_hand.urdf")
    plant.WeldFrames(
        frame_on_parent_F=plant.world_frame(),
        frame_on_child_M=plant.GetFrameByName("panda_link0"),
        X_FM=xyz_rpy_deg([0, 0, 0], [0, 0, 0]),
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
        finger_width,
        finger_width,
    ]
    prog.AddQuadraticErrorCost(np.identity(len(q)), q0_guess, q)
    prog.SetInitialGuess(q, q0_guess)
    result = Solve(ik.prog())
    soln = result.GetSolution(q)
    return soln

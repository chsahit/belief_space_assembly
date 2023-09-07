from typing import Dict

import numpy as np
from pydrake.all import (
    Expression,
    MathematicalProgram,
    RigidTransform,
    RigidTransform_,
    Solve,
)

import components
import state
from planning import motion_sets


def compute_compliance_frame(
    X_GM: RigidTransform,
    CF_d: components.ContactState,
    X_MP_dict: Dict[str, RigidTransform],
) -> RigidTransform:
    """Computes compliance frame X_GC for a desired contact CF_d.

    Computes a pose p_MC that minimizes expected torques about desired contacts. This
    is then re-rexpressed relative to the gripper by defining X_MC = (I_3, p_MC) and
    then X_GC = X_GM * X_MC. X_GM is some nominal grasp transform.

    Args:
        X_GM: RigidTransform describing a hypothesized manipuland grasp.
        CF_d: desired ContactState for the motion being refined.
        X_MP_dict: mapping from points on the manipuland which appear in CF_d to
        their pose on the the block.

    Returns:
        Compliance frame X_GC which tries to minimize the rotational displacement induced
        by linear errors about the contacts in CF_d.
    """

    def e3(i):
        v = np.zeros((3,))
        v[i] = 1
        return v

    prog = MathematicalProgram()
    p_MC = prog.NewContinuousVariables(3, "p_MC")
    X_MC = RigidTransform_[Expression](p_MC)
    for _, corner in CF_d:
        X_MP = RigidTransform_[Expression](X_MP_dict[corner].GetAsMatrix4())
        r = X_MC.translation() - X_MP.translation()
        for i in range(3):
            torque = np.cross(r, e3(i))
            prog.AddCost(torque.dot(torque))

    result = Solve(prog)
    assert result.is_success()
    X_MC_star = RigidTransform(result.GetSolution(p_MC))
    X_GC = X_GM.multiply(X_MC_star)
    return X_GC


def compliance_search(
    X_GC: RigidTransform, CF_d: components.ContactState, p: state.Particle
) -> np.ndarray:
    K_opt = components.stiff
    U_opt = motion_sets.grow_motion_set(X_GC, K_opt, CF_d, p)
    for i in range(6):
        K_curr = K_opt.copy()
        K_curr[i] = components.soft[i]
        U_curr = motion_sets.grow_motion_set(X_GC, K_curr, CF_d, p)
        if len(U_curr) > len(U_opt):
            K_opt = K_curr
            U_opt = U_curr
    return K_opt

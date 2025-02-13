import state
import utils
from simulation import ik_solver


def init_peg(
    X_GM_x: float = 0.0,
    X_GM_z: float = 0.0,
    pitch: float = 0.0,
    mu: float = 0.45,
    yaw: float = 0.0,
    y: float = 0.0,
    peg_urdf: str = "assets/peg.urdf",
) -> state.Particle:
    z = 0.16 + X_GM_z
    X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, 0.36], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([X_GM_x, 0.0, z], [180, pitch, 0])
    X_WO = utils.xyz_rpy_deg([0.5, y, 0.0], [0, 0, yaw])
    # X_WO = utils.xyz_rpy_deg([0.5, y, 0.075], [0, 0, yaw])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p0 = state.Particle(
        q_r_0,
        X_GM,
        X_WO,
        "assets/real_hole.sdf",
        peg_urdf,
        mu=mu,
    )
    return p0


def init_puzzle(
    X_GM_x: float = 0.0,
    X_GM_z: float = 0.0,
    pitch: float = 0.0,
    mu: float = 0.2,
    y: float = 0.0,
) -> state.Particle:
    z = 0.09 + X_GM_z
    X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, 0.32], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([X_GM_x, 0.0, z], [180, pitch, 0])
    X_WO = utils.xyz_rpy_deg([0.5, y, 0.01], [0, 0, 0])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p0 = state.Particle(
        q_r_0,
        X_GM,
        X_WO,
        "assets/big_fixed_puzzle.sdf",
        "assets/moving_puzzle.sdf",
        mu=mu,
    )
    return p0

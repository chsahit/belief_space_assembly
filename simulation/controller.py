from typing import Dict, Tuple

import numpy as np
from pydrake.all import BasicVector, JacobianWrtVariable, LeafSystem, RigidTransform

import components


class ControllerSystem(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.plant = plant
        self.panda = plant.GetModelInstanceByName("panda")
        self.plant_context = plant.CreateDefaultContext()
        self.panda_start_pos = plant.GetJointByName("panda_joint1").position_start()
        self.panda_end_pos = plant.GetJointByName("panda_joint7").position_start()

        self._state_port = self.DeclareVectorInputPort("state", BasicVector(18))
        self.DeclareVectorOutputPort("joint_torques", BasicVector(9), self.CalcOutput)
        self.contacts = frozenset()
        self.constraints = None
        self.sdf = dict()
        self.motion = None

    def compute_error(self, X_WC: RigidTransform, X_WCd: RigidTransform) -> np.ndarray:
        R_CCd = X_WC.InvertAndCompose(X_WCd).rotation()
        H_CCd = R_CCd.ToQuaternion()
        H_CCd = np.array([H_CCd.w(), H_CCd.x(), H_CCd.y(), H_CCd.z()])
        rot_err = (X_WC.rotation().matrix()) @ H_CCd[1:]
        xyz_err = X_WCd.translation() - X_WC.translation()
        err = np.concatenate((rot_err, xyz_err))
        return err

    def tau(
        self,
        tau_g: np.ndarray,
        J: np.ndarray,
        block_vel: np.ndarray,
        X_WG: RigidTransform,
    ) -> np.ndarray:
        if self.motion is None:
            return -tau_g
        X_WC = X_WG.multiply(self.motion.X_GC)
        err = self.compute_error(X_WC, self.motion.X_WCd)
        spring_force = np.multiply(self.motion.K, err)
        damping_force = np.multiply(self.motion.B, block_vel)
        # damping_force = 0 * np.multiply(self.motion.B, block_vel[:-1])
        # df_b = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        # df = np.multiply(df_b, block_vel)
        tau_controller = -tau_g + J.T @ (spring_force - damping_force)  # - df
        return tau_controller

    def CalcOutput(self, context, output):
        q = self._state_port.Eval(context)
        self.plant.SetPositionsAndVelocities(self.plant_context, self.panda, q)

        W = self.plant.world_frame()
        G = self.plant.GetBodyByName("panda_hand").body_frame()
        X_WG = self.plant.CalcRelativeTransform(
            self.plant_context, self.plant.world_frame(), G
        )
        J_g = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context,
            JacobianWrtVariable.kQDot,
            G,
            self.motion.X_GC.translation(),
            W,
            W,
        )
        J_g = J_g[:, self.panda_start_pos : self.panda_end_pos + 1]
        assert J_g.shape == (6, 7)
        tau_g = self.plant.CalcGravityGeneralizedForces(self.plant_context)[:7]
        block_velocity = self.plant.EvalBodySpatialVelocityInWorld(
            self.plant_context, self.plant.GetBodyByName("base_link")
        )
        block_velocity = np.concatenate(
            (block_velocity.rotational(), block_velocity.translational())
        )
        # block_velocity = J_g @ q[9:16]
        # block_velocity = q[9:16]

        tau_controller = self.tau(tau_g, J_g, block_velocity, X_WG)
        tau_controller = np.concatenate((tau_controller, np.array([5.0, 5.0])))
        output.SetFromVector(tau_controller)

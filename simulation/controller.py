import pickle
from typing import Dict, Tuple

import numpy as np
from pydrake.all import BasicVector, JacobianWrtVariable, LeafSystem, RigidTransform

import components
import mr


class ControllerSystem(LeafSystem):
    def __init__(self, plant, panda_name: str, block_name: str, out_size: int = 7):
        LeafSystem.__init__(self)
        self.plant = plant
        self.out_size = out_size
        self.panda = plant.GetModelInstanceByName(panda_name)
        self.block = plant.GetModelInstanceByName(block_name)
        self.plant_context = plant.CreateDefaultContext()
        self.panda_start_pos = plant.GetJointByName(
            "panda_joint1", self.panda
        ).position_start()
        self.panda_end_pos = plant.GetJointByName(
            "panda_joint7", self.panda
        ).position_start()

        self._state_port = self.DeclareVectorInputPort("state", BasicVector(18))
        self.contacts = frozenset()
        self.constraints = None
        self.sdf = dict()
        self.motion = None
        self.DeclareVectorOutputPort(
            "joint_torques", BasicVector(out_size), self.CalcOutput
        )
        self.i = 0
        self.history = []
        self.printed = False

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

        # print(f"{block_vel=}")
        X_WC = X_WG.multiply(self.motion.X_GC)
        block_vel_G = mr.Adjoint(X_WG.inverse().GetAsMatrix4()) @ block_vel
        err = self.compute_error(X_WC, self.motion.X_WCd)
        spring_force = np.multiply(self.motion.K, err)
        damping_force = np.multiply(self.motion.B, block_vel)
        F_C = spring_force - damping_force
        Adj_X_CG = mr.Adjoint(self.motion.X_GC.inverse().GetAsMatrix4())
        F_G = Adj_X_CG.T @ F_C
        tau_controller = -tau_g + J.T @ F_G
        return tau_controller

    def CalcOutput(self, context, output):
        q = self._state_port.Eval(context)
        self.plant.SetPositionsAndVelocities(self.plant_context, self.panda, q)

        W = self.plant.world_frame()
        G = self.plant.GetBodyByName("panda_hand", self.panda).body_frame()
        X_WG = self.plant.CalcRelativeTransform(
            self.plant_context, self.plant.world_frame(), G
        )
        if self.motion is None:
            # if not self.printed:
            # print("warning, X_GC is none")
            X_GC = RigidTransform()
            self.printed = True
        else:
            X_GC = self.motion.X_GC
        J_g = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context,
            JacobianWrtVariable.kQDot,
            G,
            X_GC.translation(),
            W,
            W,
        )
        J_g = J_g[:, self.panda_start_pos : self.panda_end_pos + 1]
        assert J_g.shape == (6, 7)
        tau_g = self.plant.CalcGravityGeneralizedForces(self.plant_context)
        # by convention, wrench vectors and twist vectors are ordered the same
        tau_g = self.plant.GetVelocitiesFromArray(self.panda, tau_g)[:7]
        block_velocity = self.plant.EvalBodySpatialVelocityInWorld(
            self.plant_context, self.plant.GetBodyByName("base_link", self.block)
        )
        block_velocity = np.concatenate(
            (block_velocity.rotational(), block_velocity.translational())
        )
        # block_velocity = np.zeros((6, ))
        # block_velocity = J_g @ q[9:16]
        # block_velocity = q[9:16]

        tau_controller = self.tau(tau_g, J_g, block_velocity, X_WG)
        tau_controller = np.zeros((self.out_size,))
        # self.history.append((context.get_time(), X_WG.translation()))
        """
        if self.i == 0:
            from pydrake.all import MultibodyForces

            to_dump = (J_g, tau_controller + tau_g)
            with open("control_logs.pkl", "wb") as f:
                pickle.dump(to_dump, f)
        """
        self.i += 1
        # tau_controller = np.concatenate((tau_controller, np.array([5.0, 5.0])))
        output.SetFromVector(tau_controller)

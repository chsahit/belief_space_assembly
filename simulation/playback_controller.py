import numpy as np
from pydrake.all import BasicVector, JacobianWrtVariable, LeafSystem

from simulation import ik_solver


class PlaybackController(LeafSystem):
    def __init__(self, plant, panda_name: str):
        LeafSystem.__init__(self)
        self.plant = plant
        self.motion = None
        self.K_q = None
        self.panda = plant.GetModelInstanceByName(panda_name)
        self.panda_start_pos = plant.GetJointByName(
            "panda_joint1", self.panda
        ).position_start()
        self.panda_end_pos = plant.GetJointByName(
            "panda_joint7", self.panda
        ).position_start()
        self.plant_context = plant.CreateDefaultContext()
        self._state_port = self.DeclareVectorInputPort("state", BasicVector(18))
        self.DeclareVectorOutputPort("joint_torques", BasicVector(7), self.CalcOutput)
        self.err = 0

    def CalcOutput(self, context, output):
        q = self._state_port.Eval(context)
        self.plant.SetPositionsAndVelocities(self.plant_context, self.panda, q)
        q_r = self.plant.GetPositions(self.plant_context, self.panda)[:7]
        q_r_dot = self.plant.GetVelocities(self.plant_context, self.panda)[:7]
        tau_g = self.plant.CalcGravityGeneralizedForces(self.plant_context)
        tau_g = self.plant.GetVelocitiesFromArray(self.panda, tau_g)[:7]
        if self.motion.q_d is None:
            m = self.motion
            X_WGd = m.X_WCd.multiply(m.X_GC.inverse())
            m.q_d = ik_solver.gripper_to_joint_states(X_WGd)[:7]
        if self.K_q is None:
            self.update_stiffness(self.motion.K, self.motion.X_GC)
        B_q = 10 * np.sqrt(self.K_q)
        self.motion.q_d = self.motion.q_d[:7]
        F_spring = np.multiply(self.K_q, self.motion.q_d - q_r)
        F_damper = np.multiply(B_q, np.zeros((7,)) - q_r_dot)
        tau_err = F_spring + F_damper
        tau_net = self.clip(tau_err - tau_g)
        self.err = tau_net
        output.SetFromVector(tau_net)

    def update_stiffness(self, K_EE, X_GC):
        J_g = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context,
            JacobianWrtVariable.kQDot,
            self.plant.GetBodyByName("panda_hand", self.panda).body_frame(),
            np.array([0, 0, 0]),
            self.plant.world_frame(),
            self.plant.world_frame(),
        )
        J_g = J_g[:, self.panda_start_pos : self.panda_end_pos + 1]
        self.K_q = np.diag(J_g.T @ np.diag(K_EE) @ J_g)

    def clip(self, torque):
        limits = np.array([87.0, 87, 87, 87, 12, 12, 12.0])
        return np.clip(torque, -limits, limits)

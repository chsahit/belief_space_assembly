from typing import Dict, Tuple

import numpy as np
from pydrake.all import BasicVector, JacobianWrtVariable, LeafSystem, RigidTransform

import components
import mr

q0 = [
    0.0796904,
    0.18628879,
    -0.07548908,
    -2.42085905,
    0.06961755,
    2.52396334,
    0.6796144,
]


class ControllerSystem(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.plant = plant
        self.panda = plant.GetModelInstanceByName("panda")
        self.plant_context = plant.CreateDefaultContext()
        self.panda_start_pos = plant.GetJointByName("panda_joint1").position_start()
        self.panda_end_pos = plant.GetJointByName("panda_joint7").position_start()

        self._state_port = self.DeclareVectorInputPort("state", BasicVector(18))
        self._acceleration_port = self.DeclareVectorInputPort("acceleration", BasicVector(9))

        self.DeclareVectorOutputPort("joint_torques", BasicVector(9), self.CalcOutput)
        self.contacts = frozenset()
        self.constraints = None
        self.sdf = dict()
        self.motion = None
        self.i = 0
        self.last_t = 0
        self.last_q_dot = None
        np.set_printoptions(precision=4, suppress=True)

    def E(self, quat: np.ndarray) -> np.ndarray:
        eta = quat[0]
        epsilon = quat[1:]
        return eta * np.eye(3) - mr.VecToso3(epsilon)

    def compute_error(self, X_WC: RigidTransform, X_WCd: RigidTransform) -> np.ndarray:
        R_CCd = X_WC.InvertAndCompose(X_WCd).rotation()
        H_CCd = R_CCd.ToQuaternion()
        H_CCd = np.array([H_CCd.w(), H_CCd.x(), H_CCd.y(), H_CCd.z()])
        rot_err = (X_WC.rotation().matrix()) @ H_CCd[1:]
        xyz_err = X_WCd.translation() - X_WC.translation()
        err = np.concatenate((rot_err, xyz_err))
        return err

    def K_o(
        self, X_WC: RigidTransform, X_WCd: RigidTransform, K: np.ndarray
    ) -> np.ndarray:
        R_CCd = X_WC.InvertAndCompose(X_WCd).rotation()
        H_CCd = R_CCd.inverse().ToQuaternion()
        H_CCd = np.array([H_CCd.w(), H_CCd.x(), H_CCd.y(), H_CCd.z()])
        K_o_local = np.diag(K[:3])
        return K_o_local
        # return 2 * (self.E(H_CCd).T @ K_o_local)

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
        linear_spring_force = np.multiply(self.motion.K[3:], err[3:])
        torsional_spring_force = (
            self.K_o(X_WC, self.motion.X_WCd, self.motion.K) @ err[:3]
        )
        spring_force = np.concatenate((torsional_spring_force, linear_spring_force))
        damping_force = np.multiply(self.motion.B, block_vel)
        total_force = spring_force - damping_force
        if np.linalg.norm(total_force) > 20:
            total_force *= (20/np.linalg.norm(total_force))

        tau_controller = -tau_g + J.T @ total_force
        return tau_controller

    def ns_stabilization(self, q_full: np.ndarray, J: np.ndarray) -> np.ndarray:
        ns_map = np.eye(7) - (np.linalg.pinv(J) @ J)
        Kp_j = 900
        Kd_j = 30
        q = q_full[:7]
        q_dot = q_full[9:16]
        return ns_map @ (Kp_j * (q0 - q) - Kd_j * q_dot)

    def print_logs(self, q, X_WG, J_g, block_vel, tau, coriolis, Mv):
        X_WC = X_WG.multiply(self.motion.X_GC)
        err = self.compute_error(X_WC, self.motion.X_WCd)
        qdot = q[9:16]
        print(f"current q: {q[:7]})")
        print(f"current qdot: {qdot}")
        print(f"current compliance frame position: {X_WC.translation()}")
        print(f"current manipuland velocity: {block_vel}")
        print(f"deisred spatial wrench: {err}")
        print(f"coriolis terms: {coriolis[:7]}")
        print(f"interial terms: {Mv}")
        print(f"output torques: {tau}")
        # print(f"expected next spatial vel = {J_g @ qdot}") # wrong, compute qdot by integrating tau
        print("-------------------------\n")

    def CalcOutput(self, context, output):
        q = self._state_port.Eval(context)
        vdot  = self._acceleration_port.Eval(context)
        self.plant.SetPositionsAndVelocities(self.plant_context, self.panda, q)

        W = self.plant.world_frame()
        G = self.plant.GetBodyByName("panda_hand").body_frame()
        X_WG = self.plant.CalcRelativeTransform(
            self.plant_context, self.plant.world_frame(), G
        )
        p_BoBp_B = self.motion.X_GC.translation()
        J_g = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context, JacobianWrtVariable.kV, G, p_BoBp_B, W, W
        )
        J_g = J_g[:, self.panda_start_pos : self.panda_end_pos + 1]
        assert J_g.shape == (6, 7)
        tau_g = self.plant.CalcGravityGeneralizedForces(self.plant_context)[:7]
        block_velocity = self.plant.EvalBodySpatialVelocityInWorld(
            self.plant_context, self.plant.GetBodyByName("panda_hand")
        )
        block_velocity = np.concatenate(
            (block_velocity.rotational(), block_velocity.translational())
        )
        tau_controller = self.tau(tau_g, J_g, block_velocity, X_WG)
        coriolis = self.plant.CalcBiasTerm(self.plant_context)
        M = self.plant.CalcMassMatrix(self.plant_context)
        if self.i == 0:
            Mv = np.zeros((7,))
        else:
            vdot = (q[9:16] - self.last_q_dot)/(context.get_time() - self.last_t)
            Mv = M[:7, :7] @ vdot[:7]
        self.last_t = context.get_time()
        self.last_q_dot = q[9:16]
        tau_controller += coriolis[:7]
        # tau_controller += Mv[:7]
        self.print_logs(q, X_WG, J_g, block_velocity, tau_controller, coriolis, Mv)

        # tau_controller += self.ns_stabilization(q, J_g)
        tau_controller = np.concatenate((tau_controller, np.array([5.0, 5.0])))
        self.i += 1
        output.SetFromVector(tau_controller)


class VirtualSpringDamper(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.plant = plant
        self.panda = plant.GetModelInstanceByName("panda")
        self.plant_context = plant.CreateDefaultContext()
        self.panda_start_pos = plant.GetJointByName("panda_joint1").position_start()
        self.panda_end_pos = plant.GetJointByName("panda_joint7").position_start()

        self._state_port = self.DeclareVectorInputPort("state", BasicVector(18))
        self._acceleration_port = self.DeclareVectorInputPort("acceleration", BasicVector(9))

        self.DeclareVectorOutputPort("vd_d", BasicVector(9), self.CalcOutput)
        self.motion = None
        self.i = 0

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
        total_force = spring_force - damping_force
        if np.linalg.norm(total_force) > 20:
            total_force *= (20/np.linalg.norm(total_force))
        if self.i % 1000 == 0:
            print(f"{err=}")
            print(f"{total_force=}")
        return J.T @ total_force

    def CalcOutput(self, context, output):
        q = self._state_port.Eval(context)
        self.plant.SetPositionsAndVelocities(self.plant_context, self.panda, q)
        W = self.plant.world_frame()
        G = self.plant.GetBodyByName("panda_hand").body_frame()
        X_WG = self.plant.CalcRelativeTransform(
            self.plant_context, self.plant.world_frame(), G
        )
        J_g = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context, JacobianWrtVariable.kV, G, [0, 0, 0], W, W
        )
        J_g_s = J_g[:, self.panda_start_pos : self.panda_end_pos + 1]
        if self.i % 1000 == 0:
            print(f"{np.linalg.cond(J_g_s)=}")
        block_velocity = self.plant.EvalBodySpatialVelocityInWorld(
            self.plant_context, self.plant.GetBodyByName("panda_hand")
        )
        block_velocity = np.concatenate(
            (block_velocity.rotational(), block_velocity.translational())
        )
        tau_controller = self.tau(None, J_g, block_velocity, X_WG)
        if self.i % 1000 == 0:
            print(f"{X_WG.translation()=}")

        M = self.plant.CalcMassMatrix(self.plant_context)
        vd_d = np.linalg.pinv(M) @ tau_controller
        output.SetFromVector(vd_d)
        self.i += 1


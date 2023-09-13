from typing import Dict, Tuple

import numpy as np
from pydrake.all import (
    AbstractValue,
    BasicVector,
    HPolyhedron,
    JacobianWrtVariable,
    LeafSystem,
    QueryObject,
    RigidTransform,
    VPolytope
)

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
        self.geom_port = self.DeclareAbstractInputPort(
            "geom_query", AbstractValue.Make(QueryObject())
        )
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
        tau_controller = -tau_g + J.T @ (spring_force - damping_force)
        return tau_controller

    def _set_constraints(self, query_obj, inspector):
        self.constraints = dict()
        for g_id in inspector.GetAllGeometryIds():
            name = inspector.GetName(g_id)
            if "bin_model" in name:
                polyhedron = HPolyhedron(VPolytope(query_obj, g_id))
                self.constraints[name] = (polyhedron.A(), polyhedron.b())

    def get_collision_set(
        self, query_object, sg_inspector
    ) -> Tuple[components.ContactState, Dict[components.Contact, float]]:
        contact_state = []
        penetrations = query_object.ComputePointPairPenetration()
        for penetration in penetrations:
            name_A = sg_inspector.GetName(penetration.id_A)
            name_B = sg_inspector.GetName(penetration.id_B)
            if "panda" in (name_A + name_B) or "Box" in (name_A + name_B):
                continue
            contact_state.append((name_A, name_B))

        try:
            sdf_data = query_object.ComputeSignedDistancePairwiseClosestPoints(0.05)
        except Exception as e:
            sdf_data = []  # GJK crashes sometimes :(
        sdf = dict()
        for dist in sdf_data:
            name_A = sg_inspector.GetName(dist.id_A)
            name_B = sg_inspector.GetName(dist.id_B)
            if ("bin_model" in name_A) and ("block" in name_B):
                sdf[(name_A, name_B)] = dist.distance
        return frozenset(contact_state), sdf

    def CalcOutput(self, context, output):
        q = self._state_port.Eval(context)
        self.plant.SetPositionsAndVelocities(self.plant_context, self.panda, q)
        query_object = self.geom_port.Eval(context)
        if self.constraints is None:
            self._set_constraints(query_object, query_object.inspector())
        self.contacts, self.sdf = self.get_collision_set(
            query_object, query_object.inspector()
        )

        W = self.plant.world_frame()
        G = self.plant.GetBodyByName("panda_hand").body_frame()
        X_WG = self.plant.CalcRelativeTransform(
            self.plant_context, self.plant.world_frame(), G
        )
        J_g = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context, JacobianWrtVariable.kQDot, G, [0, 0, 0], W, W
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

        tau_controller = self.tau(tau_g, J_g, block_velocity, X_WG)
        tau_controller = np.concatenate((tau_controller, np.array([5.0, 5.0])))
        output.SetFromVector(tau_controller)

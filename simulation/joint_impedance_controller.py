from typing import Dict, Tuple

import numpy as np
from pydrake.all import BasicVector, LeafSystem, RigidTransform

import components
from simulation import ik_solver


class JointImpedanceController(LeafSystem):
    def __init__(self, plant, panda_name):
        LeafSystem.__init__(self)
        self.plant = plant
        self.panda = plant.GetModelInstanceByName(panda_name)
        self.plant_context = plant.CreateDefaultContext()
        self._state_port = self.DeclareVectorInputPort("state", BasicVector(18))
        self.DeclareVectorOutputPort("q_d", BasicVector(14), self.CalcOutput)
        self.DeclareVectorOutputPort("gravity_ff", BasicVector(7), self.CalcGravity)
        self.motion = None
        self.q_d = None

    def CalcOutput(self, context, output):
        assert self.motion is not None
        if self.q_d is None:
            X_WGd = self.motion.X_WCd.multiply(self.motion.X_GC.inverse())
            self.q_d = ik_solver.gripper_to_joint_states(X_WGd)
            self.motion.q_d = self.q_d
        output.SetFromVector(np.append(self.q_d[:7], np.zeros((7,))))

    def CalcGravity(self, context, output):
        q = self._state_port.Eval(context)
        self.plant.SetPositionsAndVelocities(self.plant_context, self.panda, q)
        tau_g = self.plant.CalcGravityGeneralizedForces(self.plant_context)
        # by convention, wrench vectors and twist vectors are ordered the same
        tau_g = self.plant.GetVelocitiesFromArray(self.panda, tau_g)[:7]
        output.SetFromVector(-tau_g)

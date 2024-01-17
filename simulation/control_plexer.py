import numpy as np
from pydrake.all import BasicVector, LeafSystem

from simulation import ik_solver


class JointStiffnessPlexer(LeafSystem):
    def __init__(self, num_inputs: int):
        LeafSystem.__init__(self)
        self.input_ports = []
        for i in range(num_inputs):
            input_name = f"torque_{i}"
            ctrl_input_port = self.DeclareVectorInputPort(f"ctrl_{i}", BasicVector(7))
            self.input_ports.append(ctrl_input_port)
        self.DeclareVectorOutputPort("joint_torques", BasicVector(7), self.CalcOutput)

    def CalcOutput(self, context, output):
        ctrl_num = min(int(context.get_time() / 5), len(self.input_ports) - 1)
        ctrl_val = self.input_ports[ctrl_num].Eval(context)
        output.SetFromVector(ctrl_val)

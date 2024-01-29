import numpy as np
from pydrake.all import BasicVector, LeafSystem, MultibodyForces, Value, ValueProducer


class JointStiffnessController(LeafSystem):
    def __init__(self, plant, kp):
        LeafSystem.__init__(self)
        self.plant = plant
        self.kp = kp

        num_states = self.plant.num_multibody_states()
        self.num_q = self.plant.num_positions()
        print(f"{self.num_q=}")

        self.input_port_index_estimated_state_ = self.DeclareVectorInputPort(
            "estimated_state", num_states
        ).get_index()
        self.input_port_index_desired_state_ = self.DeclareVectorInputPort(
            "desired_state", BasicVector(num_states)
        ).get_index()
        self.output_port_index_force_ = self.DeclareVectorOutputPort(
            "generalized_force",
            BasicVector(self.num_q),
            self.CalcOutputForce,
            {
                self.all_input_ports_ticket(),
            },
        ).get_index()

        self.plant_context = plant.CreateDefaultContext()
        self.pc_value = Value(self.plant_context)

        self.plant_context_cache_index_ = self.DeclareCacheEntry(
            "plant_context_cache",
            ValueProducer(allocate=self.pc_value.Clone, calc=self.calc_cache),
            {
                self.input_port_ticket(
                    self.get_input_port_estimated_state().get_index()
                ),
            },
        ).cache_index()

        self.applied_forces_cache_index_ = self.DeclareCacheEntry(
            "applied_forces_cache",
            ValueProducer(
                allocate=Value(MultibodyForces(self.plant)).Clone,
                calc=self.CalcMultibodyForces,
            ),
            {
                self.cache_entry_ticket(self.plant_context_cache_index_),
            },
        ).cache_index()

    def get_input_port_estimated_state(self):
        return self.get_input_port(self.input_port_index_estimated_state_)

    def get_input_port_desired_state(self):
        return self.get_input_port(self.input_port_index_desired_state_)

    def get_output_port_generalized_force(self):
        return self.get_output_port(self.output_port_index_force_)

    def calc_cache(self, context, abstract_value):
        state = self.get_input_port_estimated_state().Eval(context)
        self.plant.SetPositionsAndVelocities(abstract_value.get_mutable_value(), state)
        # self.plant_context = abstract_value.get_mutable_value()

    def CalcOutputForce(self, context, output):
        plant_context = self.get_cache_entry(self.plant_context_cache_index_).Eval(
            context
        )
        applied_forces = self.get_cache_entry(self.applied_forces_cache_index_).Eval(
            context
        )
        tau = self.plant.CalcInverseDynamics(
            plant_context, np.zeros((self.num_q,)), applied_forces
        )
        Cv = self.plant.CalcBiasTerm(plant_context)
        tau -= Cv

        x = self.get_input_port_estimated_state().Eval(context)
        x_d =  self.get_input_port_desired_state().Eval(context)
        tau += self.kp @ (x_d[:self.num_q] - x[:self.num_q])

        output.SetFromVector(tau)


    def CalcMultibodyForces(self, context, cache_val):
        plant_context = self.get_cache_entry(self.plant_context_cache_index_).Eval(
            context
        )
        self.plant.CalcForceElementsContribution(
            plant_context, cache_val.get_mutable_value()
        )


class FixedVal(LeafSystem):
    def __init__(self, setpoint):
        LeafSystem.__init__(self)
        self.output_port_xd = self.DeclareVectorOutputPort(
            "out", BasicVector(18), self.CalcOuput
        )
        self.setpoint = setpoint

    def CalcOuput(self, context, output):
        x_d = np.concatenate((self.setpoint, np.zeros((9,))))
        output.SetFromVector(x_d)


if __name__ == "__main__":
    from pydrake.all import AddMultibodyPlantSceneGraph, DiagramBuilder

    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, 0.01)
    plant.Finalize()
    JointStiffnessController(plant, None)

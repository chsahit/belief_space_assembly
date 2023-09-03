from typing import List

from pydrake.all import Simulator

import components
from belief import belief_state


def simulate(
    p: belief_state.Particle, motion: components.CompliantMotion, vis: bool = False
) -> belief_state.Particle:
    diagram = p.make_plant(vis=vis)
    plant = diagram.GetSubsystemByName("plant")
    simulator = Simulator(diagram)
    plant_context = plant.GetMyContextFromRoot(simulator.get_mutable_context())
    controller = diagram.GetSubsystemByName("controller")
    controller.motion = motion
    simulator.Initialize()
    if vis:
        meshcat_vis = diagram.GetSubsystemByName("meshcat_visualizer(visualizer)")
        meshcat_vis.StartRecording()
        simulator.AdvanceTo(motion.timeout)
        meshcat_vis.PublishRecording()
        input()
    else:
        simulator.AdvanceTo(motion.timeout)
    q_r_T = plant.GetPositions(plant_context, plant.GetModelInstanceByName("panda"))
    p_next = p.deepcopy()
    p_next.q_r = q_r_T
    return p_next


def f_cspace(
    p: belief_state.Particle, U: List[components.CompliantMotion], multi: bool
) -> belief_state.Particle:
    pass


def f_bel(
    b: belief_state.Belief, u: components.CompliantMotion, multi: bool
) -> belief_state.Belief:
    pass

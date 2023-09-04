import multiprocessing
import signal
import sys
from typing import List, Tuple

from pydrake.all import Simulator

import components
from belief import belief_state


def simulate(
    p: belief_state.Particle, motion: components.CompliantMotion, vis: bool = False
) -> belief_state.Particle:
    """Simulates a compliant motion on a particle.

    Initializes a multibodyplant with state corresponding to the given particle and
    the controller initialized with the data from the motion. Runs the simulator until
    motion timeout and returns a particle corresponding to the final state of the plant.

    Args:
        p: The initial world configuration of type belief_state.Particle.
        motion: A CompliantMotion to be rendered by the controller.
        vis: If True, render the sim in Meshcat and wait for keyboard input before returning

    Returns:
        A particle with the same grasp and object pose hypothesis as the input but
        with new robot joint angles corresponding to the result of the motion.

    """
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


def _parallel_simulate(
    simulation_args: List[Tuple[belief_state.Particle, components.CompliantMotion]]
) -> List[belief_state.Particle]:
    # throw in a bunch of signal flags so the dump when I hit ctr+C is less messy
    p = multiprocessing.Pool(
        10, initializer=signal.signal, initargs=(signal.SIGINT, signal.SIG_IGN)
    )
    try:
        resulting_particles = p.starmap(simulate, simulation_args)
        p.close()
        p.join()
    except KeyboardInterrupt:
        print("f_cspace interrupted. Exiting")
        sys.exit()
    return resulting_particles


def f_cspace(
    p: belief_state.Particle, U: List[components.CompliantMotion], multi: bool = True
) -> List[belief_state.Particle]:
    """Wraps dynamics.py:simulate to sim many different motions on a particle in parallel.

    This function is meant for trying many candidate motions on the same initial state.
    It does NOT implement running a trajectory of motions on an initial state. That
    requires chaining multiple calls to f_cspace.
    """
    if not multi:  # might want this case later for debugging purposes
        raise NotImplementedError

    args = [(p, u) for u in U]
    return _parallel_simulate(args)


def f_bel(
    b: belief_state.Belief, u: components.CompliantMotion, multi: bool = True
) -> belief_state.Belief:
    """Wraps dynamics.py:simulate to simulate a motion on many particles in parallel.

    This function computes a posterior belief distribution conditioned on
    the prior belief "b" and a CompliantMotion "u" taken by the robot.
    """
    if not multi:  # might want this case later for debugging purposesa
        raise NotImplementedError

    args = [(p, u) for p in b.particles]
    posterior_particles = _parallel_simulate(args)
    return belief_state.Belief(posterior_particles)

import multiprocessing
import signal
import sys
import time
from typing import List, Tuple

import numpy as np
from pydrake.all import MathematicalProgram, Simulator, Solve

import components
import mr
import state

_time_in_sim = 0.0
_num_posteriors = 0


def reset_time():
    global _time_in_sim
    _time_in_sim = 0.0


def get_time():
    return _time_in_sim


def add_time(delta):
    global _time_in_sim
    _time_in_sim += delta


def reset_posteriors():
    global _num_posteriors
    _num_posteriors = 0


def get_posterior_count():
    return _num_posteriors


def add_posteriors(n):
    global _num_posteriors
    _num_posteriors += n


def tf_impedance(K_q, x0):
    samples = [x0]
    """
    for i in range(7):
        new_sample_up = np.copy(x0)
        new_sample_up[i] += 0.2
        new_sample_down = np.copy(x0)
        new_sample_down -= 0.2
        samples.append(new_sample_up)
        samples.append(new_sample_down)
    """
    mp = MathematicalProgram()
    w = mp.NewContinuousVariables(7, "w")
    print(f"{K_q @ x0[:7]=}")
    for sample in samples:
        err_vec = np.diag(w) @ sample[:7] - K_q @ sample[:7]
        mp.AddCost(err_vec.dot(err_vec))
    mp.AddBoundingBoxConstraint(0, 1000, w)
    soln = Solve(mp)
    w_star = soln.GetSolution()
    print(f"{soln.get_optimal_cost()=}")
    print(f"{w_star=}")
    return w_star


def AdvanceToWithTimeout(
    simulator: Simulator, sim_timeout: float, clock_timeout: float = 60.0
):
    curr_time_sim = 0.0
    start_time_clock = time.time()
    while curr_time_sim < sim_timeout:
        curr_time_sim += 0.1
        if time.time() - start_time_clock > clock_timeout:
            break
        simulator.AdvanceTo(curr_time_sim)


def simulate(
    p: state.Particle, motion: components.CompliantMotion, vis: bool = False
) -> state.Particle:
    """Simulates a compliant motion on a particle.

    Initializes a multibodyplant with state corresponding to the given particle and
    the controller initialized with the data from the motion. Runs the simulator until
    motion timeout and returns a particle corresponding to the final state of the plant.

    Args:
        p: The initial world configuration of type state.Particle.
        motion: A CompliantMotion to be rendered by the controller.
        vis: If True, render the sim in Meshcat and wait for keyboard input before returning

    Returns:
        A particle with the same grasp and object pose hypothesis as the input but
        with new robot joint angles corresponding to the result of the motion.
    """
    K_G = mr.Adjoint(motion.X_GC.inverse().GetAsMatrix4()) @ np.diag(motion.K)
    gains = (p.J.T) @ K_G @ p.J
    motion.is_joint_space = True
    diagram, meshcat = p.make_plant(vis=vis, gains=gains)
    plant = diagram.GetSubsystemByName("plant")
    simulator = Simulator(diagram)
    plant_context = plant.GetMyContextFromRoot(simulator.get_mutable_context())
    plant.SetPositions(plant_context, plant.GetModelInstanceByName("panda"), p.q_r)

    controller = diagram.GetSubsystemByName("controller")
    controller.motion = motion
    simulator.Initialize()
    if vis:
        meshcat_vis = diagram.GetSubsystemByName("meshcat_visualizer(visualizer)")
        meshcat_vis.StartRecording()
        try:
            # simulator.AdvanceTo(motion.timeout)
            AdvanceToWithTimeout(simulator, motion.timeout)
        except Exception as e:
            print(f"EXCEPTION: {e}")
            print(f"{motion.X_WCd=}, {motion.X_GC}")
            return None
        meshcat_vis.PublishRecording()
        with open("meshcat_html.html", "w") as f:
            f.write(meshcat.StaticHtml())
    else:
        # simulator.AdvanceTo(motion.timeout)
        AdvanceToWithTimeout(simulator, motion.timeout)
    q_r_T = plant.GetPositions(plant_context, plant.GetModelInstanceByName("panda"))
    p_next = p.deepcopy()
    p_next.q_r = q_r_T
    # p_next.trajectory = controller.history
    p_next._update_contact_data()
    return p_next


def _parallel_simulate(
    simulation_args: List[Tuple[state.Particle, components.CompliantMotion]]
) -> List[state.Particle]:
    p_sim_start = time.time()
    num_workers = min(multiprocessing.cpu_count(), len(simulation_args))
    # throw in a bunch of signal flags so the dump when I hit ctr+C is less messy
    p = multiprocessing.Pool(
        num_workers,
        initializer=signal.signal,
        initargs=(signal.SIGINT, signal.SIG_IGN),
    )
    try:
        resulting_particles = p.starmap(simulate, simulation_args, chunksize=1)
        p.close()
        p.join()
    except KeyboardInterrupt:
        print("f_cspace interrupted. Exiting")
        sys.exit()
    add_time(time.time() - p_sim_start)
    add_posteriors(len(simulation_args))
    return resulting_particles


def f_cspace(
    p: state.Particle, U: List[components.CompliantMotion], multi: bool = True
) -> List[state.Particle]:
    """Wraps dynamics.py:simulate to sim many different motions on a particle in parallel.

    This function is meant for trying many candidate motions on the same initial state.
    It does NOT implement running a trajectory of motions on an initial state. That
    requires calling simulate_trajectory.
    """
    if not multi:  # might want this case later for debugging purposes
        raise NotImplementedError

    args = [(p, u) for u in U]
    return _parallel_simulate(args)


def f_bel(
    b: state.Belief, u: components.CompliantMotion, multi: bool = True
) -> state.Belief:
    """Wraps dynamics.py:simulate to simulate a motion on many particles in parallel.

    This function computes a posterior belief distribution conditioned on
    the prior belief "b" and a CompliantMotion "u" taken by the robot.
    """
    if not multi:  # might want this case later for debugging purposesa
        raise NotImplementedError

    args = [(p, u) for p in b.particles]
    posterior_particles = _parallel_simulate(args)
    for p in posterior_particles:
        if p is None:
            print("forward pass failed, returning noop")
            return b
    return state.Belief(posterior_particles)


def parallel_f_bel(
    b: state.Belief, U: List[components.CompliantMotion]
) -> List[state.Belief]:
    jobs = []
    for u in U:
        for p in b.particles:
            jobs.append((p, u))
    job_results = _parallel_simulate(jobs)
    beliefs = []
    i_start = 0
    for idx in range(len(U)):
        i_end = i_start + len(b.particles)
        beliefs.append(state.Belief(job_results[i_start:i_end]))
        i_start = i_end
        assert len(beliefs[-1].particles) == len(b.particles)
    assert len(beliefs) == len(U)
    return beliefs


def visualize_trajectory(
    p: state.Particle,
    U: List[components.CompliantMotion],
    name: str = "meshcat_html.html",
):
    diagram, meshcat = p.make_plant(vis=True)
    plant = diagram.GetSubsystemByName("plant")
    simulator = Simulator(diagram)
    plant_context = plant.GetMyContextFromRoot(simulator.get_mutable_context())
    controller = diagram.GetSubsystemByName("controller")
    simulator.Initialize()
    meshcat_vis = diagram.GetSubsystemByName("meshcat_visualizer(visualizer)")
    meshcat_vis.StartRecording()
    t_boundary = 0.0
    for i, u in enumerate(U):
        controller.motion = u
        t_boundary += u.timeout
        try:
            simulator.AdvanceTo(t_boundary)
        except Exception as e:
            print(f"EXCEPTION: {e}")
            print(f"{motion.X_WCd=}, {motion.X_GC}")
            return None
    meshcat_vis.PublishRecording()
    with open(name, "w") as f:
        f.write(meshcat.StaticHtml())

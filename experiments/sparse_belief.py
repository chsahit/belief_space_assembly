from typing import List, Tuple

import numpy as np

import contact_defs
import dynamics
import state
import utils
from experiments import init_particle
from planning import cobs, ao_b_est

rng = np.random.default_rng(0)


def rejection_sample(
    num_samples: int,
    rangex: Tuple[float, float],
    rangez: Tuple[float, float],
    rangep: Tuple[float, float],
) -> List[state.Particle]:
    samples = []
    a = rangex[1] - rangex[0]
    b = rangez[1] - rangez[0]
    c = rangep[1] - rangep[0]
    while len(samples) < num_samples:
        x = rng.uniform(*rangex)
        z = rng.uniform(*rangez)
        pitch = rng.uniform(*rangep)
        lhs = (x**2 / a**2) + (z**2 / b**2) + (pitch**2 / c**2)
        if lhs <= 1:
            samples.append(init_particle.init_peg(X_GM_x=x, X_GM_z=z, pitch=pitch))
    return samples


def sample_belief(num_addl: int, full_belief: state.Belief) -> state.Belief:
    extrema = full_belief.particles[:7]
    non_extrema = np.array([p for p in full_belief.particles[7:]])
    rng.shuffle(non_extrema)
    return state.Belief(extrema + non_extrema[:num_addl].tolist())


def make_full_belief() -> state.Belief:
    repr_particle = init_particle.init_peg()
    x_range = [-0.01, 0.01]
    z_range = [-0.005, 0.005]
    pitch_range = [-3, 3]
    extrema = [init_particle.init_peg(X_GM_x=x_range[0]), repr_particle]
    extrema.append(init_particle.init_peg(X_GM_x=x_range[1]))
    for z in z_range:
        extrema.append(init_particle.init_peg(X_GM_z=z))
    for pitch in pitch_range:
        extrema.append(init_particle.init_peg(pitch=pitch))
    b_full_particles = []
    b_full_particles.extend(extrema)
    b_full_particles.extend(rejection_sample(8, x_range, z_range, pitch_range))
    return state.Belief(b_full_particles)


def test(num_addl: int) -> Tuple[Tuple[float, float], float]:
    g = contact_defs.peg_goal
    b_full = make_full_belief()
    times = []
    successes = []
    coarse_sr = []
    num_experiments = 20
    for i in range(num_experiments):
        b = sample_belief(num_addl, b_full)
        plan_result = cobs.cobs(b, g)
        # plan_result = ao_b_est.b_est(b, g)
        times.append(plan_result.total_time)
        if plan_result.traj is None or len(plan_result.traj) == 0:
            print("warninng, no plan found")
            successes.append(0)
            continue
        else:
            test_full = dynamics.sequential_f_bel(b_full, plan_result.traj)[-1]
        tot = sum([p.satisfies_contact(g) for p in test_full.particles])
        if test_full.satisfies_contact(g):
            coarse_sr.append(1.0)
        else:
            coarse_sr.append(0.0)
        successes.append(float(tot) / len(b_full.particles))

    stats_time = utils.median_mad(np.array(times))
    sr = utils.median_mad(np.array(successes))
    coarse_sr = sum(coarse_sr)/num_experiments
    return stats_time, sr, coarse_sr


def run_experiments():
    for num_addl in range(9):
        stats_time, stats_succ, coarse_sr = test(num_addl)
        print(f"{num_addl=}, {stats_time=}, {stats_succ=}, {coarse_sr=}")


if __name__ == "__main__":
    run_experiments()

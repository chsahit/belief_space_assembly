import components
import contact_defs
import puzzle_contact_defs
import sampler
import state
import visualize
from experiments import init_particle


def sample_from_cs_peg():
    p = init_particle.init_peg()
    visualize_samples(p, contact_defs.chamfer_init, n=32)


def sample_from_cs_puzzle():
    p = init_particle.init_puzzle()
    visualize_samples(p, puzzle_contact_defs.side)


def visualize_samples(p: state.Particle, CF_d: components.ContactState, n: int = 1):
    X_WGs = sampler.sample_from_contact(p, CF_d, num_samples=n)
    visualize.visualize_targets(p, X_WGs)


if __name__ == "__main__":
    sample_from_cs_puzzle()
    input()

from __future__ import annotations

import random
from typing import List

import numpy as np
from pydrake.all import RigidTransform, System

from simulation import plant_builder


class Particle:
    def __init__(
        self,
        q_r: np.ndarray,
        X_GB: RigidTransform,
        X_WO: RigidTransform,
        env_geom: str,
        manip_geom: str,
    ):
        self.q_r = q_r
        self.X_GB = X_GB
        self.X_WO = X_WO
        self.env_geom = env_geom
        self.manip_geom = manip_geom
        self._contacts = None
        self._constraints = None

    def make_plant(self, vis: bool = False) -> System:
        return plant_builder.make_plant(
            self.q_r, self.X_GB, self.X_WO, self.env_geom, self.manip_geom, vis=vis
        )

    @property
    def contacts(self):
        if self._contacts is not None:
            return self._contacts
        raise NotImplementedError

    @property
    def constraints(self):
        if self._constraints is not None:
            return self._constraints
        raise NotImplementedError

    def deepcopy(self) -> Particle:
        new_p = Particle(self.q_r, self.X_GB, self.X_WO, self.env_geom, self.manip_geom)
        new_p._constraints = self._constraints
        return new_p


class Belief:
    def __init__(self, particles: List[Particle]):
        self.particles = particles

    def sample(self) -> Particle:
        return random.choice(self.particles)

    @staticmethod
    def make_particles(
        G_noise: List[float], O_noise: List[float], nominal: Particle
    ) -> Belief:
        num_particles = len(G_noise[0])
        for noise in G_noise + O_noise:
            assert len(noise) == num_particles
        particles = []
        for i in range(num_particles):
            X_BG = utils.xyz_rpy_deg(
                [G_noise[0][i], 0, G_noise[1][i]], [0, G_noise[2][i], 0]
            )
            X_WO = utils.xyz_rpy_deg(
                [O_noise[0][i], O_noise[1][i], 0], [0, 0, O_noise[2][i]]
            )
            particles.append(
                Particle(nominal.q_r, X_BG, X_WO, nominal.env_geom, nominal.manip_geom)
            )
        return Belief(particles)

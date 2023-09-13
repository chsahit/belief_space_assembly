from __future__ import annotations

import random
from itertools import product
from typing import Dict, List

import numpy as np
from pydrake.all import HPolyhedron, RigidTransform, Simulator, System, VPolytope

import components
from simulation import plant_builder


class Particle:
    def __init__(
        self,
        q_r: np.ndarray,
        X_GM: RigidTransform,
        X_WO: RigidTransform,
        env_geom: str,
        manip_geom: str,
        mu: float = 0.0,
    ):
        self.q_r = q_r
        self.X_GM = X_GM
        self.X_WO = X_WO
        self.env_geom = env_geom
        self.manip_geom = manip_geom
        self.mu = mu
        self._contacts = None
        self._sdf = None
        self._constraints = None
        self._X_WG = None

    def make_plant(self, vis: bool = False, collision: bool = False) -> System:
        return plant_builder.make_plant(
            self.q_r,
            self.X_GM,
            self.X_WO,
            self.env_geom,
            self.manip_geom,
            vis=vis,
            collision_check=collision,
            mu=self.mu,
        )

    def _update_contact_data(self):
        diagram = self.make_plant(collision=True)
        controller = diagram.GetSubsystemByName("controller")
        simulator = Simulator(diagram)
        simulator.Initialize()
        simulator.AdvanceTo(0.01)
        self._contacts = controller.contacts
        self._sdf = controller.sdf

    @property
    def contacts(self) -> components.ContactState:
        if self._contacts is None:
            self._update_contact_data()
        return self._contacts

    @property
    def sdf(self) -> Dict[components.Contact, float]:
        if self._sdf is None:
            self._update_contact_data()
        return self._sdf

    @property
    def constraints(self):
        if self._constraints is not None:
            return self._constraints
        self._constraints = dict()
        diagram = self.make_plant()
        scene_graph = diagram.GetSubsystemByName("scene_graph")
        sg_query_port = scene_graph.get_query_output_port()
        query_obj = sg_query_port.Eval(scene_graph.CreateDefaultContext())
        inspector = query_obj.inspector()
        for g_id in inspector.GetAllGeometryIds():
            name = inspector.GetName(g_id)
            m = inspector.maybe_get_hydroelastic_mesh(g_id)
            if (m is None) or ("panda" in name):
                continue
            vertices = []
            X_OF = inspector.GetPoseInFrame(g_id)
            for v in m.vertices():
                X_FC = RigidTransform(v)
                X_WC = self.X_WO.multiply(X_OF).multiply(X_FC)
                vertices.append(X_WC.translation())
            vertices = np.array(vertices)
            polyhedron = HPolyhedron(VPolytope(vertices.T))
            self._constraints[name] = (polyhedron.A(), polyhedron.b())
        return self._constraints

    @property
    def X_WG(self):
        if self._X_WG is None:
            diagram = self.make_plant()
            plant = diagram.GetSubsystemByName("plant")
            plant_context = plant.GetMyContextFromRoot(diagram.CreateDefaultContext())
            plant.SetPositions(
                plant_context, plant.GetModelInstanceByName("panda"), self.q_r
            )
            self._X_WG = plant.CalcRelativeTransform(
                plant_context,
                plant.world_frame(),
                plant.GetBodyByName("panda_hand").body_frame(),
            )
        return self._X_WG

    def deepcopy(self) -> Particle:
        new_p = Particle(
            self.q_r.copy(),
            self.X_GM,
            self.X_WO,
            self.env_geom,
            self.manip_geom,
            mu=self.mu,
        )
        new_p._constraints = self._constraints
        return new_p

    def satisfies_contact(self, CF_d: components.ContactState, epsilon=0.001) -> bool:
        sdf = self.sdf
        for contact in CF_d:
            if sdf.get(contact, 0.1) > epsilon:
                return False
        return True


class Belief:
    def __init__(self, particles: List[Particle]):
        self.particles = particles

    def sample(self) -> Particle:
        return random.choice(self.particles)

    def satisfies_contact(self, CF_d: components.ContactState, epsilon=0.001) -> bool:
        for p in self.particles:
            if not p.satisfies_contact(CF_d, epsilon=epsilon):
                return False
        return True

    @staticmethod
    def make_particles(
        grasps: List[components.Grasp],
        O_poses: List[components.ObjectPose],
        nominal: Particle,
    ) -> Belief:
        assert len(grasps) == len(O_poses)
        particles = []
        for i in range(len(grasps)):
            X_GM = grasps[i].get_tf()
            X_WO = O_poses[i].get_tf()
            particles.append(
                Particle(
                    nominal.q_r,
                    X_GM,
                    X_WO,
                    nominal.env_geom,
                    nominal.manip_geom,
                    mu=nominal.mu,
                )
            )
        return Belief(particles)

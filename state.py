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
    ):
        self.q_r = q_r
        self.X_GM = X_GM
        self.X_WO = X_WO
        self.env_geom = env_geom
        self.manip_geom = manip_geom
        self._contacts = None
        self._sdf = None
        self._constraints = None

    def make_plant(
        self, vis: bool = False, collision: bool = False, ret_meshcat: bool = False
    ) -> System:
        return plant_builder.make_plant(
            self.q_r,
            self.X_GM,
            self.X_WO,
            self.env_geom,
            self.manip_geom,
            vis=vis,
            collision_check=collision,
            ret_meshcat=ret_meshcat,
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
            vertices = []
            try:
                shape = inspector.GetShape(g_id).size()
            except:
                continue
            for sgn in product([1, -1], [1, -1], [1, -1]):
                X_OF = inspector.GetPoseInFrame(g_id)
                X_FC = RigidTransform([0.5 * sgn[i] * shape[i] for i in range(3)])
                X_WC = self.X_WO.multiply(X_OF).multiply(X_FC)
                vertices.append(X_WC.translation())
            vertices = np.array(vertices)
            polyhedron = HPolyhedron(VPolytope(vertices.T))
            self._constraints[name] = (polyhedron.A(), polyhedron.b())
        return self._constraints

    def deepcopy(self) -> Particle:
        new_p = Particle(
            self.q_r.copy(), self.X_GM, self.X_WO, self.env_geom, self.manip_geom
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
                Particle(nominal.q_r, X_GM, X_WO, nominal.env_geom, nominal.manip_geom)
            )
        return Belief(particles)

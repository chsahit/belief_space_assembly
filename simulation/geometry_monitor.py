import numpy as np
from pydrake.all import (
    AbstractValue,
    BasicVector,
    EventStatus,
    HPolyhedron,
    LeafSystem,
    MinkowskiSum,
    MultibodyPlant,
    QueryObject,
    RigidTransform,
    SceneGraphInspector,
    VPolytope,
)

import utils


class GeometryMonitor(LeafSystem):
    """Perform geometric queries on a plant without having to step a simulator.

    This class has a forcedpublishevent that allows it to pull contactstate
    and geometric constraints from the plant it is connected to.
    """

    def __init__(self, plant: MultibodyPlant):
        LeafSystem.__init__(self)
        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()
        self.constraints = None
        self.manip_poly = dict()
        self.contacts = frozenset()
        self.name_to_gid = dict()
        self.sdf = dict()
        self._geom_port = self.DeclareAbstractInputPort(
            "geom_query", AbstractValue.Make(QueryObject())
        )
        # self._state_port = self.DeclareVectorInputPort("state", BasicVector(18))
        self.DeclareForcedPublishEvent(self.inspect_geometry)

    def _set_constraints(self, query_obj: QueryObject, inspector: SceneGraphInspector):
        self.constraints = dict()
        for g_id in inspector.GetAllGeometryIds():
            name = inspector.GetName(g_id)
            self.name_to_gid[name] = g_id
            if ("bin_model" in name) or ("fixed_puzzle" in name):
                polyhedron = HPolyhedron(VPolytope(query_obj, g_id))
                self.constraints[name] = (polyhedron.A(), polyhedron.b())
                self.aa_compute_fine_geometries(
                    name, self.constraints, polyhedron.A(), polyhedron.b()
                )
            if "block" in name and (not name[-3:].isnumeric()):
                frame_id_local = inspector.GetFrameId(g_id)
                body = self.plant.GetBodyFromFrameId(frame_id_local)
                frame_id_body = self.plant.GetBodyFrameIdOrThrow(body.index())
                polyhedron = HPolyhedron(
                    VPolytope(query_obj, g_id, reference_frame=frame_id_body),
                )
                # breakpoint()
                self.manip_poly[name] = (polyhedron.A(), polyhedron.b())
                self.aa_compute_fine_geometries(
                    name, self.manip_poly, polyhedron.A(), polyhedron.b()
                )
        if self.manip_poly is None:
            print("warning, manipulator geometry not cached")

    def _set_contacts(self, query_obj: QueryObject, inspector: SceneGraphInspector):
        sdf = dict()
        contact_state = []
        try:
            sdf_data = query_obj.ComputeSignedDistancePairwiseClosestPoints(0.05)
        except Exception as e:  # sometimes GJK likes to crash :(
            print("GJK crash :(")
            sdf_data = []
        for dist in sdf_data:
            name_A = inspector.GetName(dist.id_A)
            name_B = inspector.GetName(dist.id_B)
            env_cr = ("bin_model" in name_A) or ("fixed_puzzle" in name_A)
            contact_relevant = env_cr and ("block" in name_B)
            if (dist.distance < 0.0) and contact_relevant:
                contact_state.append((name_A, name_B))
            if contact_relevant:
                sdf[(name_A, name_B)] = dist.distance

        self.contacts = frozenset(contact_state)
        self.sdf = sdf

    def inspect_geometry(self, context):
        query_obj = self._geom_port.Eval(context)
        # q = self._state_port.Eval(context)
        # self.plant.SetPositionsAndVelocities(self.plant_context, q)
        inspector = query_obj.inspector()
        self._set_constraints(query_obj, inspector)
        self._set_contacts(query_obj, inspector)
        self._compute_cspace_contacts(self.plant_context)
        return EventStatus.Succeeded()

    def aa_compute_fine_geometries(self, name: str, mapping_dict, A, b):
        x_hat = np.array([1, 0, 0])
        y_hat = np.array([0, 1, 0])
        z_hat = np.array([0, 0, 1])
        descriptors = []  # (x+, x-, y+, y-, z+, z-)
        for i, n in enumerate([x_hat, y_hat, z_hat]):
            for sgn in [1, -1]:
                normal = sgn * n
                for j in range(A.shape[0]):
                    if np.linalg.norm(A[j] - normal) < 1e-5:
                        descriptors.append(sgn * b[j])

        dirs = {
            "top": (5, 4, -1),
            "bottom": (4, 5, 1),
            "front": (1, 0, -1),  # x_min should become x_max - epsilon
            "back": (0, 1, 1),
            "right": (3, 2, -1),
            "left": (2, 3, 1),  # y_max should become y_min + epsilon
        }

        A_local = np.array([x_hat, -x_hat, y_hat, -y_hat, z_hat, -z_hat])
        if A.shape[0] < 6:
            return None
        for direction, mods in dirs.items():
            local_name = name + "_" + direction
            b_local = np.copy(np.array(descriptors))
            b_local[mods[0]] = b_local[mods[1]] + (mods[2] * 1e-3)
            # convert <= inequalities to >= inequalities for mins
            b_local[1] *= -1
            b_local[3] *= -1
            b_local[5] *= -1
            mapping_dict[local_name] = (A_local, b_local)

    def _compute_cspace_contacts(self, context):
        def relevant(name: str) -> bool:
            dirs = ["top", "bottom", "front", "back", "right", "left"]
            return any([direc in name for direc in dirs])

        cspace_sdf = dict()
        for env_poly_name in self.constraints.keys():
            for m_poly_name in self.manip_poly.keys():
                if (not relevant(env_poly_name)) or (not relevant(m_poly_name)):
                    continue
                X_WO = self.plant.CalcRelativeTransform(
                    context,
                    self.plant.world_frame(),
                    self.plant.GetBodyByName("base_link").body_frame(),
                )
                m_poly_A, m_poly_b = self.manip_poly[m_poly_name]
                m_poly_A = m_poly_A @ X_WO.rotation().matrix()
                m_poly_A = -1 * m_poly_A
                env_poly = HPolyhedron(*self.constraints[env_poly_name])
                env_poly = env_poly.Scale(1.01)
                minkowski_sum = MinkowskiSum(HPolyhedron(m_poly_A, m_poly_b), env_poly)
                if minkowski_sum.PointInSet(X_WO.translation()):
                    print(f"contact in pair {(env_poly_name, m_poly_name)}")
                    cspace_sdf[(env_poly_name, m_poly_name)] = -1
        for key in cspace_sdf.keys():
            self.sdf[key] = cspace_sdf[key]

    def general_compute_fine_geometries(self, name: str, mapping_dict, A, b):
        pass

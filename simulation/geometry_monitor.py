import numpy as np
from pydrake.all import (
    AbstractValue,
    BasicVector,
    EventStatus,
    HPolyhedron,
    LeafSystem,
    MultibodyPlant,
    QueryObject,
    SceneGraphInspector,
    VPolytope,
)


class GeometryMonitor(LeafSystem):
    """Perform geometric queries on a plant without having to step a simulator.

    This class has a forcedpublishevent that allows it to pull contactstate
    and geometric constraints from the plant it is connected to.
    """

    def __init__(self, plant: MultibodyPlant):
        LeafSystem.__init__(self)
        self.plant = plant
        self.panda = plant.GetModelInstanceByName("panda")
        self.plant_context = plant.CreateDefaultContext()
        self.constraints = None
        self.manip_poly = dict()
        self.contacts = frozenset()
        self.name_to_gid = dict()
        self.sdf = dict()
        self._geom_port = self.DeclareAbstractInputPort(
            "geom_query", AbstractValue.Make(QueryObject())
        )
        self._state_port = self.DeclareVectorInputPort("state", BasicVector(18))
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

                polyhedron = HPolyhedron(
                    query_obj, g_id, reference_frame=frame_id_local
                )
                self.manip_poly[name] = (polyhedron.A(), polyhedron.b())
                success = self.aa_compute_fine_geometries(
                    name, self.manip_poly, polyhedron.A(), polyhedron.b()
                )

        if self.manip_poly is None:
            print("warning, manipulator geometry not cached")

    def _set_contacts(self, query_obj: QueryObject, inspector: SceneGraphInspector):
        sdf = dict()
        contact_state = []
        try:
            sdf_data = query_obj.ComputeSignedDistancePairwiseClosestPoints(0.005)
        except Exception:  # sometimes GJK likes to crash :(
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
        q = self._state_port.Eval(context)
        self.plant.SetPositionsAndVelocities(self.plant_context, q)
        inspector = query_obj.inspector()
        self._set_constraints(query_obj, inspector)
        self._set_contacts(query_obj, inspector)
        self._compute_cspace_contacts(self.plant_context)
        return EventStatus.Succeeded()

    def aa_compute_fine_geometries(self, name: str, mapping_dict, A, b):
        if A.shape[0] < 6:
            self.general_compute_fine_geometries(name, mapping_dict, A, b)
            return True
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
        if len(descriptors) == 0:
            print(f"no descriptors for {name}")
            return False
        for direction, mods in dirs.items():
            local_name = name + "_" + direction
            b_local = np.copy(np.array(descriptors))
            b_local[mods[0]] = b_local[mods[1]] + (mods[2] * 1e-4)
            # convert <= inequalities to >= inequalities for mins
            b_local[1] *= -1
            b_local[3] *= -1
            b_local[5] *= -1
            mapping_dict[local_name] = (A_local, b_local)
        return True

    def _compute_cspace_contacts(self, context):
        dirs = ["top", "bottom", "front", "back", "right", "left", "inside"]
        cspace_sdf = dict()
        X_WM = self.plant.CalcRelativeTransform(
            self.plant_context,
            self.plant.world_frame(),
            self.plant.GetBodyByName("base_link").body_frame(),
        )
        for (r_ec, r_mc), dist in self.sdf.items():
            if dist > 1e-3:
                continue
            for direction_b in dirs:
                m_poly_name = r_mc + "_" + direction_b
                if m_poly_name not in self.manip_poly.keys():
                    continue
                m_poly_A, m_poly_b = self.manip_poly[m_poly_name]
                m_poly_A = m_poly_A @ X_WM.rotation().inverse().matrix()
                m_poly_A = -1 * m_poly_A
                m_poly = HPolyhedron(m_poly_A, m_poly_b)
                for direction_a in dirs:
                    env_poly_name = r_ec + "_" + direction_a
                    if env_poly_name not in self.constraints.keys():
                        continue
                    env_poly = HPolyhedron(*self.constraints[env_poly_name]).Scale(1.1)
                    if env_poly.IntersectsWith(m_poly):
                        cspace_sdf[(env_poly_name, m_poly_name)] = -1
        self.sdf.update(cspace_sdf)

    def general_compute_fine_geometries(self, name: str, mapping_dict, A, b):
        # chamfer logic
        chamfer_name = name + "_inside"
        face_A = None
        face_b = None
        for row_idx in range(A.shape[0]):
            if np.linalg.norm(A[row_idx], ord=1) > 1.001:
                face_A = np.copy(A[row_idx])
                face_b = np.copy(b[row_idx])
        assert (face_A is not None) and (face_b is not None)
        face_b -= 1e-4
        face_b *= -1
        face_A *= -1
        chamfer_face_A = np.vstack((A, face_A))
        chamfer_face_b = np.append(b, np.array([face_b]))
        mapping_dict[chamfer_name] = (chamfer_face_A, chamfer_face_b)

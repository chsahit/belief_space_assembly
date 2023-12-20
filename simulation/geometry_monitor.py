import numpy as np
from pydrake.all import (
    AbstractValue,
    EventStatus,
    HPolyhedron,
    LeafSystem,
    MultibodyPlant,
    QueryObject,
    RigidTransform,
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
        self.constraints = None
        self.manip_poly = dict()
        self.contacts = frozenset()
        self.name_to_gid = dict()
        self.sdf = dict()
        self._geom_port = self.DeclareAbstractInputPort(
            "geom_query", AbstractValue.Make(QueryObject())
        )
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

        def is_fake_geom(gname: str) -> bool:
            plane_names = ["top", "bottom", "left", "right", "front", "back"]
            return any([pname in gname for pname in plane_names])

        def base_name(gname: str) -> str:
            return gname[: gname.rfind("_")]

        def poly_to_homogeneous(
            A: np.ndarray, b: np.ndarray, X: RigidTransform
        ) -> HPolyhedron:
            A_X = A @ X.GetAsMatrix34()
            b_X = np.append(b, np.array([1]))
            return HPolyhedron(A_X, b_X)

        for mpoly, meqs in self.manip_poly.items():
            for epoly, eeqs in self.constraints.items():
                if is_fake_geom(mpoly) and is_fake_geom(epoly):
                    # TODO: dont get the pose every loop
                    g_id = self.name_to_gid[base_name(mpoly)]
                    X_WM = query_obj.GetPoseInWorld(g_id)
                    mHp = poly_to_homogeneous(*meqs, X_WM)
                    eHp = HPolyhedron(*eeqs, RigidTransform())
                    mHp = mHp.Scale(1 + 1e-6)
                    contact_surface = mHp.Intersection(eHp)
                    if not contact_surface.IsEmpty():
                        sdf[(mpoly, epoly)] = -1.0

        self.contacts = frozenset(contact_state)
        self.sdf = sdf

    def inspect_geometry(self, context):
        query_obj = self._geom_port.Eval(context)
        inspector = query_obj.inspector()
        self._set_constraints(query_obj, inspector)
        self._set_contacts(query_obj, inspector)
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
        for direction, mods in dirs.items():
            local_name = name + "_" + direction
            # if ("b4_front" in local_name) or ("b5_back" in local_name):
            # breakpoint()
            b_local = np.copy(np.array(descriptors))
            b_local[mods[0]] = b_local[mods[1]] + (mods[2] * 1e-3)
            # convert <= inequalities to >= inequalities for mins
            b_local[1] *= -1
            b_local[3] *= -1
            b_local[5] *= -1
            mapping_dict[local_name] = (A_local, b_local)

    def general_compute_fine_geometries(self, name: str, mapping_dict, A, b):
        pass

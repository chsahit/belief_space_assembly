from pydrake.all import (
    AbstractValue,
    BasicVector,
    EventStatus,
    HPolyhedron,
    LeafSystem,
    QueryObject,
    SceneGraphInspector,
    VPolytope,
)


class GeometryMonitor(LeafSystem):
    """Perform geometric queries on a plant without having to step a simulator.

    This class has forcedpublishevents that allow it to pull contactstate and geometric constraints
    from the plant it is connected to.
    """

    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.plant = plant
        self.constraints = None
        self.contacts = frozenset()
        self.sdf = dict()
        self.geom_port = self.DeclareAbstractInputPort(
            "geom_query", AbstractValue.Make(QueryObject())
        )
        self.DeclareForcedPublishEvent(self.inspect_geometry)

    def _set_constraints(self, query_obj: QueryObject, inspector: SceneGraphInspector):
        self.constraints = dict()
        for g_id in inspector.GetAllGeometryIds():
            name = inspector.GetName(g_id)
            if "bin_model" in name:
                polyhedron = HPolyhedron(VPolytope(query_obj, g_id))
                self.constraints[name] = (polyhedron.A(), polyhedron.b())

    def _set_contacts(self, query_obj: QueryObject, inspector: SceneGraphInspector):
        sdf = dict()
        contact_state = []
        try:
            sdf_data = query_obj.ComputeSignedDistancePairwiseClosestPoints(0.05)
        except Exception as e:
            sdf_data = []
        for dist in sdf_data:
            name_A = inspector.GetName(dist.id_A)
            name_B = inspector.GetName(dist.id_B)
            contact_relevant = ("bin_model" in name_A) and ("block" in name_B)
            if (dist.distance < 0.0) and contact_relevant:
                contact_state.append((name_A, name_B))
            if contact_relevant:
                sdf[(name_A, name_B)] = dist.distance
        self.contacts = frozenset(contact_state)
        self.sdf = sdf

    def inspect_geometry(self, context):
        query_obj = self.geom_port.Eval(context)
        inspector = query_obj.inspector()
        self._set_constraints(query_obj, inspector)
        self._set_contacts(query_obj, inspector)
        return EventStatus.Succeeded()

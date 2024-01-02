import numpy as np
from pydrake.all import ConvexSet, HPolyhedron, MathematicalProgram, Solve


def CalcAxisAlignedBoundingBox(cvx_set: ConvexSet):
    def make_prog(inside_pt, cvx_set, axis, direction):
        prog = MathematicalProgram()
        pt = prog.NewContinuousVariables(3, "pt")
        prog.SetInitialGuess(pt, inside_pt)
        prog.AddCost(direction * pt[axis])
        cvx_set.AddPointInSetConstraints(prog, pt)
        prog.AddBoundingBoxConstraint(-10.0, 10.0, pt)
        return prog

    init_pt = cvx_set.MaybeGetFeasiblePoint()
    bounds = [[0, 0, 0], [0, 0, 0]]
    for direction_idx, direction in enumerate([1, -1]):
        for axis in [0, 1, 2]:
            m_prog = make_prog(init_pt, cvx_set, axis, direction)
            soln = Solve(m_prog)
            soln_pt = soln.GetSolution()
            bounds[direction_idx][axis] = soln_pt[axis]

    return HPolyhedron.MakeBox(np.array(bounds[0]), np.array(bounds[1])), bounds

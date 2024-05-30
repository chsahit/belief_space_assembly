import pickle
from collections import defaultdict
from typing import List

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import trimesh
from pydrake.all import (
    CollisionFilterDeclaration,
    DiagramBuilder,
    Meshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    RigidTransform,
    Role,
    RoleAssign,
    RollPitchYaw,
    RotationMatrix,
    Simulator,
)
from trimesh import proximity

import components
import contact_defs
import cspace
import dynamics
import sampler
import state
import utils
from planning import stiffness
from simulation import ik_solver, plant_builder


# yoinked from https://github.com/mpetersen94/gcs/blob/main/reproduction/prm_comparison/helpers.py
def set_transparency_of_models(plant, model_instances, color, alpha, scene_graph):
    """Sets the transparency of the given models."""
    inspector = scene_graph.model_inspector()
    for model in model_instances:
        for body_id in plant.GetBodyIndices(model):
            frame_id = plant.GetBodyFrameIdOrThrow(body_id)
            for geometry_id in inspector.GetGeometries(frame_id, Role.kIllustration):
                properties = inspector.GetIllustrationProperties(geometry_id)
                try:
                    phong = properties.GetProperty("phong", "diffuse")
                    if color is not None:
                        phong.set(*color, alpha)
                    else:
                        phong.set(phong.r(), phong.g(), phong.b(), alpha)
                    properties.UpdateProperty("phong", "diffuse", phong)
                    scene_graph.AssignRole(
                        plant.get_source_id(),
                        geometry_id,
                        properties,
                        RoleAssign.kReplace,
                    )
                except Exception:
                    pass


def _make_combined_plant(b: state.Belief, meshcat: Meshcat):
    builder = DiagramBuilder()
    plant, scene_graph, parser = plant_builder.init_plant(builder, timestep=0.005)
    parser.SetAutoRenaming(True)
    instance_list = list()
    for i, p in enumerate(b.particles):
        parser_i = Parser(plant, "i")
        panda = parser_i.AddModels("assets/panda_arm_hand.urdf")[0]
        plant.RenameModelInstance(panda, "panda_" + str(i))
        env_geometry = parser_i.AddModels(p.env_geom)[0]
        plant.RenameModelInstance(env_geometry, "obj_" + str(i))
        manipuland = parser_i.AddModels(p.manip_geom)[0]
        plant.RenameModelInstance(manipuland, "block_" + str(i))
        plant_builder._weld_geometries(
            plant, p.X_GM, p.X_WO, panda, manipuland, env_geometry
        )
        plant_builder._set_frictions(
            plant, scene_graph, [env_geometry, manipuland], p.mu
        )
        instance_list.append((panda, env_geometry, manipuland))
    plant.Finalize()

    colors = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    for i, p in enumerate(b.particles):
        P, O, M = instance_list[i]
        set_transparency_of_models(plant, [P, O, M], colors[i % 3], 0.5, scene_graph)
        plant.SetDefaultPositions(P, p.q_r)
        plant_builder.wire_controller(
            P,
            f"controller_{str(i)}",
            f"panda_{str(i)}",
            f"setpoint_{str(i)}",
            builder,
            plant,
        )

    meshcat_vis = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat, MeshcatVisualizerParams()
    )
    diagram = builder.Build()
    manager = scene_graph.collision_filter_manager()
    for p_idx_i in range(len(b.particles)):
        P_i, O_i, M_i = instance_list[p_idx_i]
        P_i_bodies = plant.GetBodyIndices(P_i)
        O_i_bodies = plant.GetBodyIndices(O_i)
        M_i_bodies = plant.GetBodyIndices(M_i)
        geom_set_i = plant.CollectRegisteredGeometries(
            [plant.get_body(b_idx) for b_idx in (P_i_bodies + O_i_bodies + M_i_bodies)]
        )
        for p_idx_j in range(p_idx_i + 1, len(b.particles)):
            P_j, O_j, M_j = instance_list[p_idx_j]
            P_j_bodies = plant.GetBodyIndices(P_j)
            O_j_bodies = plant.GetBodyIndices(O_j)
            M_j_bodies = plant.GetBodyIndices(M_j)
            geom_set_j = plant.CollectRegisteredGeometries(
                [
                    plant.get_body(b_idx)
                    for b_idx in (P_j_bodies + O_j_bodies + M_j_bodies)
                ]
            )
            declaration = CollisionFilterDeclaration().ExcludeBetween(
                geom_set_i, geom_set_j
            )
            manager.Apply(declaration)

    return diagram


def play_motions_on_belief(
    b: state.Belief, U: List[components.CompliantMotion], fname: str = None
):
    meshcat = Meshcat()
    diagram = _make_combined_plant(b, meshcat)
    simulator = Simulator(diagram)
    visualizer = diagram.GetSubsystemByName("meshcat_visualizer(visualizer)")
    visualizer.StartRecording()
    T = 0.0
    for u in U:
        u = ik_solver.update_motion_qd(u)
        T += u.timeout
        for i in range(len(b.particles)):
            controller = diagram.GetSubsystemByName("controller_" + str(i))
            setpoint = diagram.GetSubsystemByName("setpoint_" + str(i))
            setpoint.setpoint = u.q_d
            controller.kp = u.K
        simulator.AdvanceTo(T)
    visualizer.PublishRecording()
    if fname is not None:
        with open(fname, "w") as f:
            f.write(meshcat.StaticHtml())


def show_particle(p: state.Particle):
    u = components.CompliantMotion(
        RigidTransform(), p.X_WG, components.stiff, timeout=0.0
    )
    u = ik_solver.update_motion_qd(u)
    dynamics.simulate(p, u, vis=True)


def show_benchmarks(fname: str):
    print(f"displaying: {fname}")
    import matplotlib.pyplot as plt

    with open(fname, "rb") as f:
        data = pickle.load(f)
    for dvar_idx, dvar in enumerate(["simulator_calls", "wall_time", "sim_time"]):
        trends = defaultdict(list)
        for params, results in data.items():
            stats = utils.result_statistics(results)
            mu, std = stats[dvar_idx]
            trends[params[1]].append((params[0], (mu, mu - std, mu + std)))
        for planner, trend in trends.items():
            if dvar == "wall_time":
                print(f"\n{planner=}, {dvar=}")
            x_coords = [float(stat[0]) for stat in trend]
            y_coords = [stat[1][0] for stat in trend]
            lb = [stat[1][1] for stat in trend]
            ub = [stat[1][2] for stat in trend]
            sorted_order = np.argsort(x_coords)
            xcs, ycs, lbs, ubs = list(), list(), list(), list()
            for i in range(len(x_coords)):
                xcs.append(x_coords[sorted_order[i]])
                ycs.append(y_coords[sorted_order[i]])
                lbs.append(lb[sorted_order[i]])
                ubs.append(ub[sorted_order[i]])
                if dvar == "wall_time":
                    med = np.round(ycs[-1], decimals=3)
                    mad = np.round(ubs[-1] - ycs[-1], decimals=3)
                    print(f"deviation={xcs[-1]},time=${med} \\pm {mad}$")
            plt.fill_between(xcs, lbs, ubs, alpha=0.2)
            plt.plot(xcs, ycs, label=planner)
        if "pitch" in fname:
            plt.xlabel("Amount of Uncertainty (degrees)")
        else:
            plt.xlabel("Amount of Uncertainty (meters)")
        plt.ylabel(dvar)
        plt.legend()
        plt.savefig(f"{fname[:3]}_{dvar}.png", dpi=1200)
        plt.close()


def playback_result(b, fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    # results = data[("0.009", "True")]
    results = data[("4", "True")]
    for result in results:
        if (result.traj is not None) and len(result.traj) == 4:
            play_motions_on_belief(b, result.traj)
            break
    print("done")
    input()


def visualize_targets(p_nom: state.Particle, targets: List[RigidTransform]):
    for target in targets:
        p_vis = p_nom.deepcopy()
        p_vis.q_r = ik_solver.gripper_to_joint_states(target)
        p_vis._X_WG = None
        u_noop = components.CompliantMotion(
            RigidTransform(), p_vis.X_WG, components.stiff, timeout=0.001
        )
        u_noop.q_d = p_vis.q_r
        # p_vis.env_geom = "assets/floor.sdf"
        # p_vis.X_WO = RigidTransform([0.5, 0, -1.0])
        dynamics.simulate(p_vis, u_noop, vis=True)


def render_graph(nx_graph: nx.Graph, label_dict):
    edge_x = []
    edge_y = []
    pos = nx.spring_layout(nx_graph)
    for edge in nx_graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=0.5, color="#888"), mode="lines"
    )
    node_x = [pos[node][0] for node in nx_graph.nodes()]
    node_y = [pos[node][1] for node in nx_graph.nodes()]
    labels = [label_dict[node] for node in nx_graph.nodes()]
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        marker=dict(
            showscale=False,
            colorscale="YlGnBu",
            reversescale=True,
            color=[],
            size=10,
            line_width=2,
        ),
    )
    node_trace.text = labels
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False)
    fig.write_html("mode_graph.html")


def save_trimesh(slice_2D, tf, ax, test_fns=[], cs=[]):
    from matplotlib.patches import Polygon

    ax.set_aspect("equal", "datalim")
    # hardcode a format for each entity type
    eformat = {
        "Line0": {"color": "g", "linewidth": 1},
        "Line1": {"color": "y", "linewidth": 1},
        "Arc0": {"color": "r", "linewidth": 1},
        "Arc1": {"color": "b", "linewidth": 1},
        "Bezier0": {"color": "k", "linewidth": 1},
        "Bezier1": {"color": "k", "linewidth": 1},
        "BSpline0": {"color": "m", "linewidth": 1},
        "BSpline1": {"color": "m", "linewidth": 1},
    }
    # assert rotation.IsValid()
    for entity in slice_2D.entities:
        # if the entity has it's own plot method use it
        if hasattr(entity, "plot"):
            entity.plot(slice_2D.vertices)
            continue
        # otherwise plot the discrete curve
        discrete = entity.discrete(slice_2D.vertices)
        # a unique key for entities
        e_key = entity.__class__.__name__ + str(int(entity.closed))

        fmt = eformat[e_key].copy()
        if hasattr(entity, "color"):
            # if entity has specified color use it
            fmt["color"] = "b"
            # if len(test_fns) > 0:
            #     fmt["linestyle"] = "dotted"
        xs = []
        ys = []
        pts = []
        for i in range(len(discrete.T[0])):
            coord = np.array([discrete.T[0][i], discrete.T[1][i], 0, 1])
            coord_W = tf.GetAsMatrix4() @ coord
            xs.append(coord_W[0])
            ys.append(coord_W[2])
            pts.append([coord_W[0], coord_W[2]])
        pts = np.array(pts)
        ax.add_patch(Polygon(pts, facecolor="b"))
        # ax.plot(xs, ys, **fmt)
        for c_idx, test_fn in enumerate(test_fns):
            colored_xs = []
            colored_ys = []
            for i in range(len(xs)):
                if test_fn(xs[i], ys[i]):
                    colored_xs.append(xs[i])
                    colored_ys.append(ys[i])
            fmt["color"] = cs[c_idx]
            fmt["linewidth"] = 2
            ax.plot(colored_xs, colored_ys, **fmt)
        if len(test_fns) > 0:
            ax.text(0.495, 0.07, r"$c_2$", color="m", fontsize=28)
            ax.text(0.51, 0.125, r"$c_1$", color="y", fontsize=28)


def project_to_planar(p: state.Particle, ax, u: components.CompliantMotion = None):
    R_WM_flat = np.copy(p.X_WM.rotation().ToRollPitchYaw().vector())
    R_WM_flat[0] = 0
    R_WM_flat[2] = 0
    R_WM_flat_vec = np.array(R_WM_flat)
    R_WM_flat = RotationMatrix(RollPitchYaw(R_WM_flat))
    p_WM_flat = np.array([p.X_WM.translation()[0], 0, p.X_WM.translation()[2]])

    mesh = cspace.ConstructCspaceSlice(cspace.ConstructEnv(p), R_WM_flat).mesh
    compliance_dir = stiffness.translational_normal(
        RigidTransform(R_WM_flat, p_WM_flat), mesh
    )

    # if dump_mesh:
    #     utils.dump_mesh(mesh)
    cross_section = mesh.section(
        plane_origin=np.array([0.5, 0.0, 0.0]), plane_normal=([0, 1, 0])
    )
    planar, to_3d = cross_section.to_planar()
    # print(f"{to_3d=}")
    X_Wo = RigidTransform(np.array(to_3d))
    save_trimesh(planar, X_Wo, ax)
    pose_t2 = [[p.X_WM.translation()[0], 0, p.X_WM.translation()[2]]]
    if mesh.contains(pose_t2)[0]:
        closest, _, _ = proximity.closest_point(mesh, pose_t2)
        pose_t2 = [closest[0][0], closest[0][2]]
    else:
        pose_t2 = [pose_t2[0][0], pose_t2[0][2]]
    ax.plot(*pose_t2, "ro")
    if compliance_dir is not None and u is not None:
        normed_dir = 0.025 * (compliance_dir / np.linalg.norm(compliance_dir))
        arrow_mags = [normed_dir[0], normed_dir[2]]
        ax.arrow(*pose_t2, *arrow_mags)
    if u is not None:
        u_WM = u.X_WCd.multiply(p.X_GM)
        sp = [u_WM.translation()[0], u_WM.translation()[2]]
        ax.plot(*sp, "go")
    q_M = [p_WM_flat[0], p_WM_flat[2], (180.0 * R_WM_flat_vec[1] / np.pi)]
    q_M_round = [round(x, 3) for x in q_M]
    q_M_str = r"{}".format(str(q_M_round))
    # ax.xaxis.set_visible(False)
    import matplotlib.pyplot as plt

    plt.setp(ax.spines.values(), visible=False)
    ax.set_xlabel(r"$q_M={}$".format(q_M_str))


def show_belief_space_traj(traj_fname: str):
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    with open(traj_fname, "rb") as f:
        dat = pickle.load(f)
    traj = dat["trajectory"]
    U = dat["motions"]
    fnames = []
    for i, belief in enumerate(traj):
        if i < len(U):
            u = U[i]
        else:
            u = None
        img_name = show_belief_space_step(belief, u, i)
        fnames.append(img_name)
    fnames_sched = show_contact_schedule(traj[0].particles[1])
    fnames = [fnames_sched] + fnames
    fig = plt.figure(figsize=(10, 7))
    axes = []
    num_panels = len(fnames)
    axes.append(fig.add_subplot(2, 1, 1, aspect="equal"))
    for j in range(num_panels - 1):
        # axes.append(fig.add_subplot(1, num_panels, j + 1, aspect="equal"))
        axes.append(fig.add_subplot(2, num_panels - 1, num_panels + j, aspect="equal"))
    for k, ax in enumerate(axes):
        if k == 0:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(mpimg.imread(fnames[k]))
            ax.set_xlabel("Contact Sequence")
            continue
        timestep = k - 1
        ax.set_xticks([])
        raw_t = r"{}".format(str(timestep))
        ax.set_xlabel(r"$t = {}$".format(raw_t))
        ax.set_yticks([])
        ax.imshow(mpimg.imread(fnames[k]))
    fig.tight_layout()
    fig.savefig("trajectory.eps", dpi=800)


def show_contact_schedule(p: state.Particle):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    R_WM_flat = np.copy(p.X_WM.rotation().ToRollPitchYaw().vector())
    R_WM_flat[0] = 0
    R_WM_flat[2] = 0
    R_WM_flat = RotationMatrix(RollPitchYaw(R_WM_flat))

    mesh = cspace.ConstructCspaceSlice(cspace.ConstructEnv(p), R_WM_flat).mesh

    cross_section = mesh.section(
        plane_origin=np.array([0.5, 0.0, 0.0]), plane_normal=([0, 1, 0])
    )
    planar, to_3d = cross_section.to_planar()
    X_Wo = RigidTransform(np.array(to_3d))

    def test_fn(pt_x: float, pt_z: float) -> bool:
        return pt_x > 0.501 and pt_x < 0.52 and pt_z > -0.001
        # return pt_x > 0.505 and pt_x < 0.5125 and pt_z > -0.001

    def test_fn2(pt_x, pt_z):
        return pt_z < 0.09 and pt_z > 0.079

    save_trimesh(planar, X_Wo, ax, test_fns=[test_fn, test_fn2], cs=["y", "m"])
    plt.setp(ax.spines.values(), visible=False)
    fname_saved = "contact_sched.eps"
    fig.tight_layout()
    fig.savefig(fname_saved)
    return fname_saved


def show_belief_space_step(b_curr: state.Belief, u: components.CompliantMotion, i: int):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(1, 2, 1, aspect="equal")
    ax2 = fig.add_subplot(2, 2, 2, aspect="equal")
    ax3 = fig.add_subplot(2, 2, 4, aspect="equal")
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(left=0.44, right=0.56)
        ax.set_ylim(bottom=0.07, top=0.19)
        # ax.axis("off")
    project_to_planar(b_curr.particles[1], ax1, u=u)
    project_to_planar(b_curr.particles[0], ax2)
    project_to_planar(b_curr.particles[2], ax3)
    fname_saved = f"planner_step_{i}.png"
    fig.tight_layout()
    fig.savefig(fname_saved, dpi=800)
    return fname_saved

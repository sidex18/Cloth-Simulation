import argparse
import warp as wp
import warp.render

from config           import SimConfig
from mesh             import create_cloth_grid, load_collider_mesh
from collision.response import DynamicCollider, merge_collider_meshes
from solver.integrator  import ClothSimulation

wp.init()


def run(config: SimConfig):
    # cloth mesh
    positions, is_pinned, spring_pairs, rest_lengths, stiffness, masses, triangles = \
        create_cloth_grid(config)

    # collider(s)
    collider = None
    all_paths = list(config.collider_paths)
    if config.collider_path is not None:
        all_paths.append(config.collider_path)

    if all_paths:
        mesh_list = [load_collider_mesh(p) for p in all_paths]
        if len(mesh_list) == 1:
            verts, faces = mesh_list[0]
        else:
            verts, faces = merge_collider_meshes(mesh_list)
        collider = DynamicCollider(verts, faces)

    # simulation
    sim = ClothSimulation(
        config, positions, is_pinned,
        spring_pairs, rest_lengths, stiffness, masses,
        collider=collider,
    )

    # USD renderer
    renderer = wp.render.UsdRenderer(config.out)

    # simulation loop
    num_frames = int(config.duration / config.h)
    print(f"[main] Simulating {num_frames} frames -> {config.out}")

    for frame in range(num_frames):
        sim.step()

        if renderer is not None:
            t = frame * config.h
            renderer.begin_frame(t)
            renderer.render_mesh(
                name="cloth",
                points=sim.x.numpy(),
                indices=triangles,
            )
            if sim.collider is not None:
                renderer.render_mesh(
                    name="collider",
                    points=sim.collider.mesh.points.numpy(),
                    indices=sim.collider.mesh.indices.numpy(),
                )
            renderer.end_frame()

        if frame % 30 == 0:
            print(f"[main] frame {frame:4d} / {num_frames}")

    sim.print_timing_summary()
    sim.print_convergence_summary()

    if renderer is not None:
        renderer.save()
        print(f"[main] Saved to {config.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast Mass-Spring Cloth Simulation")
    parser.add_argument("--res",      type=int,   default=32)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--out",      type=str,   default="cloth_sim.usd")
    parser.add_argument("--collider", type=str, nargs="*", default=[],
                        help="One or more collider OBJ file paths")
    parser.add_argument("--precond",  action="store_true",
                        help="Enable Jacobi diagonal preconditioning")
    args = parser.parse_args()

    cfg = SimConfig(
        res=args.res,
        duration=args.duration,
        out=args.out,
        collider_paths=args.collider,
        use_jacobi_precond=args.precond,
        pinned_vertices=[(0, 0), (0, args.res - 1)],  # top-left, top-right
    )
    run(cfg)

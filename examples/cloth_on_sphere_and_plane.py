import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warp as wp
wp.init()

from config               import SimConfig
from mesh                 import create_cloth_grid, load_collider_mesh, generate_sphere_obj, generate_plane_obj
from collision.response   import DynamicCollider, merge_collider_meshes
from solver.integrator    import ClothSimulation
import warp.render


ASSETS_DIR = os.path.join(os.path.dirname(__file__), "../assets")
SPHERE_OBJ = os.path.join(ASSETS_DIR, "sphere.obj")
PLANE_OBJ  = os.path.join(ASSETS_DIR, "plane.obj")


def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)

    if not os.path.exists(SPHERE_OBJ):
        generate_sphere_obj(SPHERE_OBJ, radius=0.3, lat_steps=24, lon_steps=24)

    if not os.path.exists(PLANE_OBJ):
        generate_plane_obj(PLANE_OBJ, size=2.0, y=-0.3, subdivisions=4)

    config = SimConfig(
        res=32,
        size=1.0,
        h=1.0 / 300.0,
        duration=2.0,
        stiffness=200.0,
        mass_total=1.0,
        num_bcd_iters=10,
        num_gd_iters=10,
        gd_alpha=53.0,
        use_jacobi_precond=True,
        collider_paths=[SPHERE_OBJ, PLANE_OBJ],
        collision_margin=0.01,
        out="cloth_on_sphere_and_plane.usd",
        pinned_vertices=[],  # cloth falls freely
    )

    # cloth mesh
    positions, is_pinned, spring_pairs, rest_lengths, stiffness, masses, triangles = \
        create_cloth_grid(config)

    # colliders — merge sphere + plane into one BVH
    mesh_list = [load_collider_mesh(p) for p in config.collider_paths]
    merged_verts, merged_faces = merge_collider_meshes(mesh_list)
    collider = DynamicCollider(merged_verts, merged_faces)

    sim = ClothSimulation(
        config, positions, is_pinned,
        spring_pairs, rest_lengths, stiffness, masses,
        collider=collider,
    )

    renderer   = wp.render.UsdRenderer(config.out)
    num_frames = int(config.duration / config.h)
    print(f"Simulating {num_frames} frames...")

    for frame in range(num_frames):
        sim.step()

        t = frame * config.h
        renderer.begin_frame(t)

        # cloth
        renderer.render_mesh(
            name="cloth",
            points=sim.x.numpy(),
            indices=triangles,
        )

        # colliders (static)
        renderer.render_mesh(
            name="colliders",
            points=merged_verts,
            indices=merged_faces.flatten().tolist(),
        )

        renderer.end_frame()

        if frame % 30 == 0:
            print(f"  frame {frame:4d} / {num_frames}")

    sim.print_timing_summary()
    sim.print_convergence_summary()
    renderer.save()
    print(f"Saved to {config.out}")


if __name__ == "__main__":
    main()

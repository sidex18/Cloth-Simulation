import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warp as wp
wp.init()

from config               import SimConfig
from mesh                 import create_cloth_grid, load_collider_mesh, generate_sphere_obj
from collision.response   import DynamicCollider, merge_collider_meshes
from solver.integrator    import ClothSimulation
import warp.render


SPHERE_OBJ = os.path.join(os.path.dirname(__file__), "../assets/sphere.obj")


def main():
    # generate sphere OBJ if not present
    os.makedirs(os.path.dirname(SPHERE_OBJ), exist_ok=True)
    if not os.path.exists(SPHERE_OBJ):
        generate_sphere_obj(SPHERE_OBJ, radius=0.3, lat_steps=24, lon_steps=24)

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
        collider_paths=[SPHERE_OBJ],
        collision_margin=0.01,
        out="cloth_on_sphere.usd",
        pinned_vertices=[],  # cloth falls freely onto sphere
    )

    # cloth mesh
    positions, is_pinned, spring_pairs, rest_lengths, stiffness, masses, triangles = \
        create_cloth_grid(config)

    # collider(s)
    mesh_list = [load_collider_mesh(p) for p in config.collider_paths]
    if len(mesh_list) == 1:
        verts, faces = mesh_list[0]
    else:
        verts, faces = merge_collider_meshes(mesh_list)
    collider = DynamicCollider(verts, faces)

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

        # collider (static, rendered once per frame for visualization)
        renderer.render_mesh(
            name="sphere",
            points=verts,
            indices=faces.flatten().tolist(),
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

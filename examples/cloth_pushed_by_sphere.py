import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warp as wp
wp.init()

from config               import SimConfig
from mesh                 import create_cloth_grid, load_collider_mesh, generate_sphere_obj
from collision.response   import DynamicCollider
from solver.integrator    import ClothSimulation
import warp.render


SPHERE_OBJ = os.path.join(os.path.dirname(__file__), "../assets/sphere.obj")


def sphere_position(t):
    """Sphere starts at y=2.0, descends at 0.3 m/s."""
    return (0.0, 1.4 - 0.3 * t, 0.0)


def main():
    # generate sphere OBJ if not present
    os.makedirs(os.path.dirname(SPHERE_OBJ), exist_ok=True)
    if not os.path.exists(SPHERE_OBJ):
        generate_sphere_obj(SPHERE_OBJ, radius=0.3, lat_steps=24, lon_steps=24)

    res = 32
    config = SimConfig(
        res=res,
        size=1.0,
        h=1.0 / 300.0,
        duration=2.5,
        stiffness=2000.0,
        mass_total=1.0,
        num_bcd_iters=10,
        num_gd_iters=10,
        gd_alpha=53.0,
        use_jacobi_precond=True,
        collider_paths=[SPHERE_OBJ],
        collision_margin=0.01,
        out="cloth_pushed_by_sphere.usd",
        pinned_vertices=[(0, 0), (0, res - 1), (res - 1, 0), (res - 1, res - 1)],
    )

    # cloth mesh
    positions, is_pinned, spring_pairs, rest_lengths, stiffness, masses, triangles = \
        create_cloth_grid(config)

    # dynamic collider — sphere descends onto pinned cloth
    verts, faces = load_collider_mesh(SPHERE_OBJ)
    collider = DynamicCollider(verts, faces, transform_fn=sphere_position)

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

        # collider (dynamic — re-read updated positions each frame)
        renderer.render_mesh(
            name="sphere",
            points=collider.mesh.points.numpy(),
            indices=collider.mesh.indices.numpy(),
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

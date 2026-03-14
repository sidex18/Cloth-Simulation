import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warp as wp
wp.init()

from config              import SimConfig
from mesh                import create_cloth_grid
from solver.integrator   import ClothSimulation
import warp.render


def main():
    res = 32
    config = SimConfig(
        res=res,
        size=1.0,
        h=1.0 / 30.0,
        duration=5.0,
        stiffness=2000.0,
        mass_total=1.0,
        num_bcd_iters=10,
        num_gd_iters=10,
        gd_alpha=0.056,
        use_jacobi_precond=False,
        collider_path=None,
        out="cloth_falling.usd",
        pinned_vertices=[(0, 0), (0, res - 1)], # pin top-left (0,0) and top-right (0, res-1) corners
    )

    positions, is_pinned, spring_pairs, rest_lengths, stiffness, masses, triangles = \
        create_cloth_grid(config)

    sim = ClothSimulation(
        config, positions, is_pinned,
        spring_pairs, rest_lengths, stiffness, masses,
        collider=None,
    )

    renderer  = wp.render.UsdRenderer(config.out)
    num_frames = int(config.duration / config.h)
    print(f"Simulating {num_frames} frames...")

    for frame in range(num_frames):
        sim.step()

        t = frame * config.h
        renderer.begin_frame(t)
        renderer.render_mesh(
            name="cloth",
            points=sim.x.numpy(),
            indices=triangles,
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

import warp as wp


@wp.kernel
def gradient_step_kernel(
    x:         wp.array(dtype=wp.vec3),
    grad:      wp.array(dtype=wp.vec3),
    A_diag:    wp.array(dtype=float),
    is_pinned: wp.array(dtype=int),
    alpha:     float,
    use_precond: int,
):
    """Per-vertex position update. Supports plain GD (use_precond=0) and Jacobi (use_precond=1). Skips pinned vertices."""
    tid = wp.tid()

    # TODO

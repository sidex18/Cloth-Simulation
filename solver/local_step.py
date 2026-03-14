import warp as wp


@wp.kernel
def local_step_kernel(
    x:             wp.array(dtype=wp.vec3),
    spring_indices:wp.array(dtype=wp.vec2i),
    rest_lengths:  wp.array(dtype=float),
    d:             wp.array(dtype=wp.vec3),
):
    """Local step: update auxiliary spring direction d for each spring."""
    tid = wp.tid()

    # TODO

import warp as wp


@wp.kernel
def compute_inertia_target(
    x:        wp.array(dtype=wp.vec3),
    v:        wp.array(dtype=wp.vec3),
    is_pinned:wp.array(dtype=int),
    h:        float,
    gravity:  wp.vec3,
    y:        wp.array(dtype=wp.vec3),
):
    """Compute the inertia target y for each vertex."""
    tid = wp.tid()

    # TODO


@wp.kernel
def eval_inertia_energy(
    x:    wp.array(dtype=wp.vec3),
    y:    wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    loss: wp.array(dtype=float),
):
    """Inertia energy term. Must be differentiable (runs inside wp.Tape)."""
    tid = wp.tid()

    # TODO


@wp.kernel
def eval_spring_energy(
    x:             wp.array(dtype=wp.vec3),
    d:             wp.array(dtype=wp.vec3),
    spring_indices:wp.array(dtype=wp.vec2i),
    stiffness:     wp.array(dtype=float),
    h:             float,
    loss:          wp.array(dtype=float),
):
    """Spring energy term. Must be differentiable (runs inside wp.Tape)."""
    tid = wp.tid()

    # TODO

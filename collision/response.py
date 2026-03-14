import numpy as np
import warp as wp


def build_collider(vertices, faces):
    """Construct a wp.Mesh BVH from collider geometry."""
    mesh = wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3),
        indices=wp.array(faces.flatten(), dtype=int),
    )
    return mesh


@wp.kernel
def translate_vertices_kernel(
    rest_points: wp.array(dtype=wp.vec3),
    offset:      wp.vec3,
    out_points:  wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    out_points[tid] = rest_points[tid] + offset


class DynamicCollider:
    """Collider that optionally moves each frame via transform_fn(t) -> (dx, dy, dz)."""

    def __init__(self, vertices, faces, transform_fn=None):
        self.mesh = build_collider(vertices, faces)
        self.rest_points = wp.array(vertices, dtype=wp.vec3)
        self.num_points = len(vertices)
        self.transform_fn = transform_fn

    def update(self, t):
        if self.transform_fn is None:
            return
        dx, dy, dz = self.transform_fn(t)
        wp.launch(
            kernel=translate_vertices_kernel,
            dim=self.num_points,
            inputs=[self.rest_points, wp.vec3(dx, dy, dz), self.mesh.points],
        )
        self.mesh.refit()

    @property
    def id(self):
        return self.mesh.id


def merge_collider_meshes(mesh_list):
    """Merge multiple (vertices, faces) pairs into a single combined mesh."""
    all_verts = []
    all_faces = []
    vertex_offset = 0

    for verts, faces in mesh_list:
        all_verts.append(verts)
        all_faces.append(faces + vertex_offset)
        vertex_offset += len(verts)

    merged_verts = np.concatenate(all_verts, axis=0).astype(np.float32)
    merged_faces = np.concatenate(all_faces, axis=0).astype(np.int32)
    print(f"[collision] Merged {len(mesh_list)} meshes -> "
          f"{len(merged_verts)} verts, {len(merged_faces)} faces")
    return merged_verts, merged_faces


@wp.kernel
def collision_kernel(
    x:         wp.array(dtype=wp.vec3),
    is_pinned: wp.array(dtype=int),
    mesh_id:   wp.uint64,
    margin:    float,
    max_dist:  float,
):
    """Project penetrating vertices outside the collider surface."""
    tid = wp.tid()

    # TODO

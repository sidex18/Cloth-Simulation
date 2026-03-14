import math
import numpy as np
import trimesh

def create_cloth_grid(config):
    """
    Generates an (res x res) cloth mesh in the XZ plane, suspended at y=1.

    Returns
    -------
    positions   : list of (x, y, z) tuples,  length = res*res
    is_pinned   : list of int (0 or 1),       length = res*res
    spring_pairs: list of (i, j) index pairs
    rest_lengths: list of floats
    stiffness   : list of floats
    masses      : list of floats
    triangles   : flat list of ints (index triples for rendering)
    """
    res = config.res
    size = config.size
    k = config.stiffness
    m_each = config.mass_total / (res * res)
    spacing = size / (res - 1)

    # vertices
    positions = []
    masses = []
    for i in range(res):
        for j in range(res):
            x = i * spacing - size / 2.0
            y = 1.0                          # suspend 1 m above origin
            z = j * spacing - size / 2.0
            positions.append((x, y, z))
            masses.append(m_each)

    # pinned vertices 
    is_pinned = [0] * (res * res)
    for (gi, gj) in config.pinned_vertices:
        is_pinned[gi * res + gj] = 1

    # springs
    spring_pairs = []
    rest_lengths = []
    stiffness = []

    def add_spring(p1, p2):
        spring_pairs.append((p1, p2))
        d = math.sqrt(sum(
            (positions[p1][dim] - positions[p2][dim]) ** 2
            for dim in range(3)
        ))
        rest_lengths.append(d)
        stiffness.append(k)

    for i in range(res):
        for j in range(res):
            idx = i * res + j
            # structural
            if i < res - 1:
                add_spring(idx, idx + res)
            if j < res - 1:
                add_spring(idx, idx + 1)
            # shear
            if i < res - 1 and j < res - 1:
                add_spring(idx,     idx + res + 1)
                add_spring(idx + 1, idx + res)

    triangles = []
    for i in range(res - 1):
        for j in range(res - 1):
            i0 = i * res + j
            i1 = i * res + j + 1
            i2 = (i + 1) * res + j
            i3 = (i + 1) * res + j + 1
            triangles.extend([i0, i2, i1,
                               i1, i2, i3])

    return positions, is_pinned, spring_pairs, rest_lengths, stiffness, masses, triangles


def generate_sphere_obj(path, radius=0.3, lat_steps=20, lon_steps=20):

    verts = []
    faces = []

    for i in range(lat_steps + 1):
        theta = math.pi * i / lat_steps          
        for j in range(lon_steps):
            phi = 2.0 * math.pi * j / lon_steps 
            x = radius * math.sin(theta) * math.cos(phi)
            y = radius * math.cos(theta)
            z = radius * math.sin(theta) * math.sin(phi)
            verts.append((x, y, z))

    for i in range(lat_steps):
        for j in range(lon_steps):
            a = i * lon_steps + j
            b = i * lon_steps + (j + 1) % lon_steps
            c = (i + 1) * lon_steps + j
            d = (i + 1) * lon_steps + (j + 1) % lon_steps
            faces.append((a, b, c))
            faces.append((b, d, c))

    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f"[mesh] Wrote sphere OBJ to {path}")


def generate_plane_obj(path, size=2.0, y=0.0, subdivisions=1):
    """
    Generate a flat plane OBJ centered at the origin at height y.

    Parameters
    ----------
    path : str
        Output OBJ file path.
    size : float
        Side length of the plane.
    y : float
        Height (Y coordinate) of the plane.
    subdivisions : int
        Number of subdivisions per axis (1 = 2 triangles, 2 = 8 triangles, etc.).
    """
    half = size / 2.0
    n = subdivisions + 1  # vertices per axis

    verts = []
    for i in range(n):
        for j in range(n):
            x = -half + size * i / subdivisions
            z = -half + size * j / subdivisions
            verts.append((x, y, z))

    faces = []
    for i in range(subdivisions):
        for j in range(subdivisions):
            a = i * n + j
            b = i * n + j + 1
            c = (i + 1) * n + j
            d = (i + 1) * n + j + 1
            faces.append((a, b, c))
            faces.append((b, d, c))

    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f"[mesh] Wrote plane OBJ to {path}")


def load_collider_mesh(obj_path):
    """
    Loads an OBJ file using trimesh and returns (vertices, faces) as numpy arrays.

    Returns
    -------
    vertices : np.ndarray, shape (V, 3), float32
    faces    : np.ndarray, shape (F, 3), int32
    """

    mesh = trimesh.load(obj_path, force="mesh")
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces,    dtype=np.int32)
    print(f"[mesh] Loaded collider: {obj_path}  "
          f"({len(vertices)} verts, {len(faces)} faces)")
    return vertices, faces

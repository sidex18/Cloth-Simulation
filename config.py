from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SimConfig:
    # mesh
    res: int = 32                   # cloth grid resolution (res x res vertices)
    size: float = 1.0               # cloth side length in meters

    # time
    h: float = 1.0 / 30.0          # timestep (s)
    duration: float = 5.0          # total simulation time (s)

    # material
    stiffness: float = 200.0       # spring stiffness (N/m)
    mass_total: float = 1.0         # total cloth mass (kg)

    # gravity
    gravity: tuple = (0.0, -9.81, 0.0)

    # solver
    num_bcd_iters: int = 10         # max BCD iterations per timestep
    num_gd_iters: int = 10          # max gradient descent steps per global step
    gd_alpha: float = 0.56              # GD step size (ignored when use_jacobi_precond=True)
    use_jacobi_precond: bool = False  # if True, scale gradient by 1/A_diag (recovers Jacobi)
    bcd_tol: float = 1e-6          # BCD early termination: relative energy change threshold
    gd_tol: float = 1e-6           # GD early termination: relative energy change threshold

    # collision
    collider_path: Optional[str] = None   # path to single OBJ file (shorthand)
    collider_paths: list = field(default_factory=list)  # list of OBJ file paths
    collision_margin: float = 0.01        # offset from collider surface (m)

    # output
    out: str = "cloth_sim.usd"

    # pinning
    # List of (i, j) grid indices to pin.
    pinned_vertices: list = field(default_factory=lambda: [])

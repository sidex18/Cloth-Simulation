**Name:** [Your Name] | **Student #:** [Your Student Number] | **CWL:** [Your CWL Username]

**Collaborators:** [Names of people you discussed with, or N/A]
**Resources:** [Websites or references you used, or N/A]
**AI tools:** [AI tools you used, or N/A]

**Notes to marker:** [Any additional information, or N/A]

---

# CPSC 426 Assignment 4: Cloth Simulation

GPU-accelerated cloth simulation using the *Fast Mass-Spring* method (Liu et al. 2013), built with [NVIDIA Warp](https://github.com/NVIDIA/warp).

For the full assignment specification, tasks, and grading breakdown, see [`A4_spec.pdf`](A4_spec.pdf).

## Setup

```bash
conda create -n cpsc426 -c conda-forge --override-channels \
    python=3.14.2 numpy pip openusd -y
conda activate cpsc426
pip install warp-lang trimesh
```

## Running

```bash
# Part I: cloth falling under gravity (no collision)
python examples/falling_cloth.py

# Part II: cloth draping onto a sphere
python examples/cloth_on_sphere.py

# Part II: dynamic collider pushing pinned cloth
python examples/cloth_pushed_by_sphere.py

# Part II: multiple colliders (sphere + plane)
python examples/cloth_on_sphere_and_plane.py

# General CLI
python main.py --res 32 --duration 5.0 --out cloth.usd
python main.py --res 32 --precond --collider assets/sphere.obj
```

Output `.usd` files can be viewed in Blender (File > Import > Universal Scene Description).

## Reference

T. Liu, A. W. Bargteil, J. F. O'Brien, and L. Kavan.
*Fast Simulation of Mass-Spring Systems.*
ACM Transactions on Graphics (SIGGRAPH Asia), 32(6), 2013.
Included as [`liu13fast.pdf`](liu13fast.pdf).

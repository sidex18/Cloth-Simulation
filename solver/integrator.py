import numpy as np
import warp as wp

from solver.energy      import compute_inertia_target, eval_inertia_energy, eval_spring_energy
from solver.local_step  import local_step_kernel
from solver.global_step import gradient_step_kernel
from collision.response import collision_kernel



@wp.kernel
def update_velocities_kernel(
    x_new:     wp.array(dtype=wp.vec3),
    x_old:     wp.array(dtype=wp.vec3),
    is_pinned: wp.array(dtype=int),
    h:         float,
    v:         wp.array(dtype=wp.vec3),
):
    """v_i = (x_new_i - x_old_i) / h.  Pinned vertices get zero velocity."""
    tid = wp.tid()
    if is_pinned[tid] == 1:
        v[tid] = wp.vec3(0.0, 0.0, 0.0)
    else:
        v[tid] = (x_new[tid] - x_old[tid]) / h



class ClothSimulation:
    def __init__(self, config, positions, is_pinned,
                 spring_pairs, rest_lengths, stiffness_vals, masses,
                 collider=None):
        self.config       = config
        self.num_particles = len(positions)
        self.num_springs   = len(spring_pairs)
        self.collider = collider

        h = config.h
        g = config.gravity

        # x has requires_grad for wp.Tape autodiff
        self.x = wp.array(positions, dtype=wp.vec3, requires_grad=True)
        self.x_old = wp.zeros(self.num_particles, dtype=wp.vec3)
        self.v = wp.zeros(self.num_particles, dtype=wp.vec3)
        self.y = wp.zeros(self.num_particles, dtype=wp.vec3)

        self.mass = wp.array(masses, dtype=float)
        self.is_pinned = wp.array(is_pinned, dtype=int)
        self.gravity = wp.vec3(g[0], g[1], g[2])

        self.spring_indices = wp.array(spring_pairs, dtype=wp.vec2i)
        self.rest_lengths = wp.array(rest_lengths, dtype=float)
        self.stiffness = wp.array(stiffness_vals, dtype=float)

        self.d = wp.zeros(self.num_springs, dtype=wp.vec3)
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)

        # -- TODO: compute self.A_diag (Jacobi diagonal preconditioner) -- #
        self.A_diag = wp.zeros(self.num_particles, dtype=float)

        # ---------------------------------------------------------------- #

        self.alpha = config.gd_alpha
        self.use_precond = 1 if config.use_jacobi_precond else 0

        self.timings = {}
        self.frame_count = 0
        self.convergence_log = []  # per-frame: list of BCD energy values
        self._frame_energies = []

        print(f"[sim] {self.num_particles} vertices, "
              f"{self.num_springs} springs, "
              f"h={h:.4f}s, alpha={self.alpha:.6f}, "
              f"precond={'Jacobi' if self.use_precond else 'none'}")

    # ------------------------------------------------------------------
    #  Provided helpers — use these in your step() implementation
    # ------------------------------------------------------------------

    def _timer(self, name):
        """Context manager for timing a phase of step().
        Usage: with self._timer("local_step"): ...
        """
        return wp.ScopedTimer(name, print=False, dict=self.timings, synchronize=True)

    def _begin_step(self):
        """Call at the start of step() to reset per-frame state."""
        self._frame_energies = []

    def _end_step(self):
        """Call at the end of step() to finalize convergence log."""
        self.convergence_log.append(self._frame_energies)
        self.frame_count += 1

    def _check_gd_convergence(self, prev_energy):
        """Check GD convergence after a gradient step.
        Returns (converged: bool, current_energy: float).
        """
        energy = self.loss.numpy()[0]
        if prev_energy != float('inf'):
            rel_change = abs(prev_energy - energy) / (abs(prev_energy) + 1e-12)
            if rel_change < self.config.gd_tol:
                return True, energy
        return False, energy

    def _check_bcd_convergence(self, prev_energy):
        """Check BCD convergence after a full local+global(+collision) cycle.
        Returns (converged: bool, current_energy: float).
        Internally tracks per-frame energy history for convergence_log.
        """
        energy = self._eval_total_energy(self.config.h)
        self._frame_energies.append(energy)
        if prev_energy != float('inf'):
            rel_change = abs(prev_energy - energy) / (abs(prev_energy) + 1e-12)
            if rel_change < self.config.bcd_tol:
                return True, energy
        return False, energy

    # ------------------------------------------------------------------

    def step(self):
        """Advance the simulation by one timestep. Implements the full BCD loop."""
        cfg = self.config
        h = cfg.h

        # TODO: implement the full BCD time-stepping pipeline (see spec Section 2).
        #
        # Helpers for timing and convergence (used by print_timing_summary /
        # print_convergence_summary):
        #   self._begin_step() / self._end_step()    — call at start / end of step
        #   self._timer(name)                         — context manager for timing a phase
        #   self._check_gd_convergence(prev_energy)   — inner GD early termination
        #   self._check_bcd_convergence(prev_energy)  — outer BCD early termination
        # Both convergence helpers return (converged: bool, current_energy: float).
        pass

    def _eval_total_energy(self, h):
        """Evaluate total energy (inertia + spring) without gradient computation."""
        self.loss.zero_()
        wp.launch(
            kernel=eval_inertia_energy,
            dim=self.num_particles,
            inputs=[self.x, self.y, self.mass, self.loss],
        )
        wp.launch(
            kernel=eval_spring_energy,
            dim=self.num_springs,
            inputs=[self.x, self.d, self.spring_indices,
                    self.stiffness, h, self.loss],
        )
        return self.loss.numpy()[0]

    def print_convergence_summary(self):
        """Print convergence info: early terminations and any unconverged frames."""
        if not self.convergence_log:
            return
        cfg = self.config
        n = len(self.convergence_log)
        early_count = sum(1 for e in self.convergence_log if len(e) < cfg.num_bcd_iters)
        max_iter_frames = [i for i, e in enumerate(self.convergence_log) if len(e) == cfg.num_bcd_iters]

        print(f"\n[convergence] {n} frames: "
              f"{early_count} converged early, "
              f"{len(max_iter_frames)} hit max BCD iters ({cfg.num_bcd_iters})")

        if max_iter_frames:
            # report worst unconverged frames by final relative change
            worst = []
            for fi in max_iter_frames:
                e = self.convergence_log[fi]
                if len(e) >= 2:
                    rel = abs(e[-1] - e[-2]) / (abs(e[-2]) + 1e-12)
                    worst.append((fi, rel))
            worst.sort(key=lambda x: -x[1])
            for fi, rel in worst[:5]:
                print(f"  frame {fi}: final rel_change = {rel:.2e} "
                      f"(energy {self.convergence_log[fi][-1]:.4e})")

    def print_timing_summary(self):
        """Print average timing per frame for each phase."""
        n = self.frame_count
        if n == 0:
            return
        print(f"\n[timing] Average over {n} frames:")
        for key in ["step", "local_step", "global_step", "collision"]:
            if key not in self.timings:
                continue
            vals = self.timings[key]
            avg = sum(vals) / len(vals)
            total = sum(vals)
            print(f"  {key:14s}  avg {avg:8.2f} ms/call  "
                  f"({len(vals):5d} calls, {total:10.1f} ms total)")

#!/usr/bin/env python3
"""
3D Coupled PDE Solver with Ricci Curvature, Extended Tracking, and Real-Time Plotting
Now updated to save unique GIFs for each (alpha, beta) combination.

This script solves a 3D PDE of the form:
   ∂²E/∂t² = c² ∇²E - alpha*c²*R(x,y,z)*E + c²*beta*cos(ωt)

Where R(x, y, z) is a Ricci-like curvature field, and alpha & beta
are coupling parameters. We optionally visualize the z-mid slice in
real-time, track max amplitude and total energy, and save final plots
only if the final amplitude changes by more than a given threshold
relative to the last saved run. Additionally, we capture frames in-memory
and create an animated GIF for each run, named based on (alpha, beta).
"""

import os
import io
import time
import logging
from typing import List, Tuple, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio  # needed for creating animated GIFs

# --------------------------------------------------------------------
# 1) Setup Logging
# --------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


# --------------------------------------------------------------------
# 2) Parameter Class
# --------------------------------------------------------------------
class CoupledParams3D:
    """
    Parameters for the 3D PDE:

    PDE:
      ∂²E/∂t² = c² ∇²E - alpha*c²*R_field*E + c²*beta*cos(ωt)

    Fields:
    - alpha, beta: PDE coupling parameters
    - c: wave speed
    - omega: driving frequency
    - nx, ny, nz: 3D domain resolution
    - dx, dy, dz: cell sizes
    - dt: time step
    - time_steps: total iteration steps
    - mass: sets scale for R(x,y,z)
    - damping: amplitude factor each iteration (e.g., 0.98 => 2% decay/step)
    - output_dir: folder for results
    - realtime_interval: how often we update the real-time slice
    - enable_realtime_plotting: toggles live plot
    - random_init: if True, uses a random initial field instead of sin(x)*sin(y)*sin(z)
    """

    def __init__(
        self,
        alpha: float = 1e-6,
        beta: float = 1e-7,
        c: float = 5e3,
        omega: float = 0.5,
        nx: int = 44,
        ny: int = 44,
        nz: int = 44,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: float = 1.0,
        dt: float = 5e-10,
        time_steps: int = 3000,
        mass: float = 5e24,
        damping: float = 0.98,
        output_dir: str = "3d_extended_run",
        realtime_interval: int = 50,
        enable_realtime_plotting: bool = True,
        random_init: bool = True
    ):
        self.alpha = alpha
        self.beta = beta
        self.c = c
        self.omega = omega
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt
        self.time_steps = time_steps
        self.mass = mass
        self.damping = damping
        self.output_dir = output_dir
        self.realtime_interval = realtime_interval
        self.enable_realtime_plotting = enable_realtime_plotting
        self.random_init = random_init


# --------------------------------------------------------------------
# 3) Ricci Curvature in 3D
# --------------------------------------------------------------------
def define_ricci_3d(params: CoupledParams3D) -> np.ndarray:
    """
    Generate a Ricci-like curvature field:
      R(x,y,z) = mass / (r^2 + smoothing) * 1e-6
    """
    x_vals = np.linspace(-params.nx/2, params.nx/2, params.nx)
    y_vals = np.linspace(-params.ny/2, params.ny/2, params.ny)
    z_vals = np.linspace(-params.nz/2, params.nz/2, params.nz)
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')

    r2 = X**2 + Y**2 + Z**2
    smoothing = 1e3

    R_field = params.mass / (r2 + smoothing)
    R_field *= 1e-6
    return R_field


# --------------------------------------------------------------------
# 4) Field Initialization
# --------------------------------------------------------------------
def initialize_field_3d(params: CoupledParams3D) -> Tuple[np.ndarray, np.ndarray]:
    """
    If random_init=True, use a small random field to break symmetry.
    Otherwise, use sin(x)*sin(y)*sin(z).

    Returns (E, E_prev) for the PDE updates.
    """
    if params.random_init:
        # For reproducible random patterns each run
        np.random.seed(42)
        E_init = 1e-10 * (np.random.rand(params.nx, params.ny, params.nz) - 0.5)
    else:
        x_vals = np.linspace(0, np.pi, params.nx)
        y_vals = np.linspace(0, np.pi, params.ny)
        z_vals = np.linspace(0, np.pi, params.nz)
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
        E_init = 1e-10 * np.sin(X) * np.sin(Y) * np.sin(Z)

    E_prev = E_init.copy()
    return E_init, E_prev


# --------------------------------------------------------------------
# 5) Laplacian in 3D
# --------------------------------------------------------------------
def laplacian_3d(E: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """
    3D finite-difference Laplacian of E.
    """
    d2E = np.zeros_like(E)

    # x-direction
    d2E[1:-1, :, :] += (E[2:, :, :] - 2*E[1:-1, :, :] + E[:-2, :, :]) / (dx**2)
    # y-direction
    d2E[:, 1:-1, :] += (E[:, 2:, :] - 2*E[:, 1:-1, :] + E[:, :-2, :]) / (dy**2)
    # z-direction
    d2E[:, :, 1:-1] += (E[:, :, 2:] - 2*E[:, :, 1:-1] + E[:, :, :-2]) / (dz**2)

    return d2E


# --------------------------------------------------------------------
# 6) Single PDE Run with Real-Time Plotting + Unique GIF per (alpha,beta)
# --------------------------------------------------------------------
def run_single_sim_realtime(
    params: CoupledParams3D,
    alpha_val: float,
    beta_val: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs the 3D PDE for 'time_steps'. Tracks:
      - max_amplitudes[t] = max |E| at step t
      - energy_over_time[t] = 0.5 * sum(E^2) at step t

    If enable_realtime_plotting=True, updates a slice every 'realtime_interval' steps.
    We capture each plotted frame by saving the figure to an in-memory PNG, then reading
    it into a frames list for an animated GIF. The GIF is named uniquely with alpha_val, beta_val.
    """
    os.makedirs(params.output_dir, exist_ok=True)

    dt2 = (params.dt)**2
    c2 = (params.c)**2
    alpha_c2 = params.alpha * c2

    # Build curvature and fields
    R_field = define_ricci_3d(params)
    E, E_prev = initialize_field_3d(params)

    max_amplitudes = np.zeros(params.time_steps, dtype=np.float64)
    energy_over_time = np.zeros(params.time_steps, dtype=np.float64)

    # This list will store each frame for the animated GIF
    frames = []

    # Real-time plotting setup
    if params.enable_realtime_plotting:
        plt.ion()
        fig, ax = plt.subplots(figsize=(5, 4))
        z_mid = params.nz // 2
        slice_data = np.abs(E[:, :, z_mid])
        heatmap = ax.imshow(slice_data, origin='lower', cmap='inferno')
        cbar = plt.colorbar(heatmap, ax=ax)
        cbar.set_label("Field |E|")
        ax.set_title("Real-Time Heatmap (z_mid slice)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    else:
        fig = None
        heatmap = None
        z_mid = params.nz // 2

    # PDE iteration
    pbar = tqdm(range(params.time_steps), desc="Sim PDE", ncols=100)
    for t in pbar:
        # PDE source
        source_val = params.beta * np.cos(params.omega * params.dt * t)
        lapl = laplacian_3d(E, params.dx, params.dy, params.dz)

        PDE_term = c2 * lapl - alpha_c2 * (R_field * E) + (c2 * source_val)
        E_new = 2.0 * E - E_prev + dt2 * PDE_term

        # Damping
        E_new *= params.damping
        # Clamp extremes
        E_new = np.clip(E_new, -1e5, 1e5)

        # Check for stability
        if not np.isfinite(E_new).all():
            logging.warning(f"NaN/Inf detected at iteration {t}, stopping.")
            pbar.close()
            break

        E_prev = E
        E = E_new

        # Track metrics
        max_amplitudes[t] = np.max(np.abs(E))
        energy_over_time[t] = 0.5 * np.sum(E**2)

        # Update real-time slice occasionally
        if params.enable_realtime_plotting and (t % params.realtime_interval == 0):
            slice_data = np.abs(E[:, :, z_mid])
            heatmap.set_data(slice_data)
            heatmap.set_clim(slice_data.min(), slice_data.max())
            plt.draw()
            plt.pause(0.001)

            # -------------------------------------------------
            # Capture the frame for GIF using an in-memory PNG
            # -------------------------------------------------
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=72)
            buf.seek(0)
            frame = imageio.imread(buf)
            frames.append(frame)
            buf.close()

    pbar.close()

    # Create a unique GIF filename based on (alpha_val, beta_val)
    if params.enable_realtime_plotting and len(frames) > 1:
        gif_name = f"simulation_a{alpha_val:.1e}_b{beta_val:.1e}.gif"
        gif_path = os.path.join(params.output_dir, gif_name)
        imageio.mimsave(gif_path, frames, fps=5)
        logging.info(f"   -> Animated GIF saved to {gif_path}")

    # Close real-time figure
    if params.enable_realtime_plotting and fig is not None:
        plt.ioff()
        plt.close(fig)

    return E, max_amplitudes, energy_over_time, R_field


# --------------------------------------------------------------------
# 7) Parameter Sweep with Thresholded Saves
# --------------------------------------------------------------------
def param_sweep_3d(
    alpha_list: List[float],
    beta_list: List[float],
    base_params: CoupledParams3D,
    save_threshold: float = 0.01
) -> List[Tuple[float, float, float, Optional[float]]]:
    """
    Sweep over alpha & beta. For each combo:
      - run PDE
      - only save final plots if final amplitude changes > 'save_threshold'
        vs. the last saved run

    Returns:
        summary: list of (alpha, beta, final_maxE, last_amp)
    """
    os.makedirs(base_params.output_dir, exist_ok=True)

    summary = []
    total_runs = len(alpha_list) * len(beta_list)
    run_count = 0
    last_saved_amp = None

    for alpha_ in alpha_list:
        for beta_ in beta_list:
            run_count += 1
            desc = f"(alpha={alpha_:.2e}, beta={beta_:.2e})"
            logging.info(f"--- [Run {run_count}/{total_runs}] PDE 3D {desc} ---")

            # Build fresh params for this run
            sim_params = CoupledParams3D(
                alpha=alpha_,
                beta=beta_,
                c=base_params.c,
                omega=base_params.omega,
                nx=base_params.nx, ny=base_params.ny, nz=base_params.nz,
                dx=base_params.dx, dy=base_params.dy, dz=base_params.dz,
                dt=base_params.dt, time_steps=base_params.time_steps,
                mass=base_params.mass,
                damping=base_params.damping,
                output_dir=base_params.output_dir,
                realtime_interval=base_params.realtime_interval,
                enable_realtime_plotting=base_params.enable_realtime_plotting,
                random_init=base_params.random_init
            )

            start_time = time.time()
            # Pass alpha_, beta_ so run_single_sim_realtime can name the GIF
            final_E, max_amps, energy_arr, _ = run_single_sim_realtime(
                sim_params, alpha_val=alpha_, beta_val=beta_
            )
            end_time = time.time()
            logging.info(f"   PDE run took {end_time - start_time:.2f} s.")

            final_maxE = np.max(np.abs(final_E))
            last_amp = max_amps[-1] if len(max_amps) else None

            # Decide if we should save final plots
            should_save = False
            if last_saved_amp is None:
                should_save = True
            else:
                rel_diff = abs(final_maxE - last_saved_amp) / max(last_saved_amp, 1e-30)
                if rel_diff > save_threshold:
                    should_save = True

            if should_save:
                logging.info(
                    f"   -> Saving results (amplitude changed > {100*save_threshold:.1f}%)"
                )

                # Slicing
                z_mid = sim_params.nz // 2
                y_mid = sim_params.ny // 2
                z_slice = np.abs(final_E[:, :, z_mid])
                y_slice = np.abs(final_E[:, y_mid, :])

                # 1) Final z-mid slice
                figz, axz = plt.subplots(figsize=(5, 4))
                axz.set_title(f"Final |E| (z={z_mid}) {desc}")
                imz = axz.imshow(z_slice, origin='lower', cmap='inferno')
                plt.colorbar(imz, ax=axz, label="|E|")
                outz = os.path.join(
                    sim_params.output_dir, f"finalZ_a{alpha_:.1e}_b{beta_:.1e}.png"
                )
                plt.savefig(outz, dpi=120)
                plt.close(figz)

                # 2) Final y-mid slice
                figy, axy = plt.subplots(figsize=(5, 4))
                axy.set_title(f"Final |E| (y={y_mid}) {desc}")
                imy = axy.imshow(y_slice.T, origin='lower', cmap='inferno')
                plt.colorbar(imy, ax=axy, label="|E|")
                outy = os.path.join(
                    sim_params.output_dir, f"finalY_a{alpha_:.1e}_b{beta_:.1e}.png"
                )
                plt.savefig(outy, dpi=120)
                plt.close(figy)

                # 3) Max|E| & Energy vs Time
                figm, axm = plt.subplots(figsize=(5, 3.5))
                axm.plot(max_amps, color='blue', label='Max |E|')
                axm.set_xlabel("Time Step")
                axm.set_ylabel("Max |E|", color='blue')
                axm.tick_params(axis='y', labelcolor='blue')

                axm2 = axm.twinx()
                axm2.plot(energy_arr, color='red', label='Energy')
                axm2.set_ylabel("Total Energy (0.5 * ΣE^2)", color='red')
                axm2.tick_params(axis='y', labelcolor='red')

                axm.set_title(f"Max|E| vs Time {desc}")
                lines1, labels1 = axm.get_legend_handles_labels()
                lines2, labels2 = axm2.get_legend_handles_labels()
                axm.legend(lines1 + lines2, labels1 + labels2, loc="best")

                outm = os.path.join(
                    sim_params.output_dir, f"maxAmpEnergy_a{alpha_:.1e}_b{beta_:.1e}.png"
                )
                plt.savefig(outm, dpi=120)
                plt.close(figm)

                last_saved_amp = final_maxE
            else:
                logging.info("   -> Skipping save (no significant amplitude change).")

            summary.append((alpha_, beta_, final_maxE, last_amp))

    return summary


# --------------------------------------------------------------------
# 8) Main: Extended Run + Param Sweep
# --------------------------------------------------------------------
def run_3d_extended_realtime_sweep() -> None:
    """
    Main function to run a parameter sweep over alpha,beta and 
    do real-time PDE simulations, saving final results if amplitude changes enough.
    Each run saves a unique GIF named based on (alpha, beta).
    """
    base_params = CoupledParams3D(
        alpha=1e-6,
        beta=1e-7,
        c=5e3,
        omega=0.5,
        nx=44, ny=44, nz=44,
        dx=0.050, dy=0.050, dz=0.050,
        dt=5e-10,
        time_steps=3000,
        mass=5e24,
        damping=0.98,
        output_dir="3d_extended_run",
        realtime_interval=50,
        enable_realtime_plotting=True,
        random_init=True
    )

    alpha_list = [1e-6, 2e-6, 3e-6, 4e-6]
    beta_list  = [1e-7, 2e-7, 3e-7, 4e-7, 5e-7, 6e-7, 7e-7, 8e-7, 9e-7, 1e-6]

    logging.info("\n--- Starting Extended 3D PDE Sweep with Real-Time Heatmap ---")
    logging.info(f"Domain: {base_params.nx}×{base_params.ny}×{base_params.nz}, "
                 f"steps={base_params.time_steps}, dt={base_params.dt}")
    logging.info(f"alpha_list={alpha_list}")
    logging.info(f"beta_list={beta_list}\n")

    global_start = time.time()
    results = param_sweep_3d(alpha_list, beta_list, base_params, save_threshold=0.01)
    global_end = time.time()

    # Print a summary
    logging.info("\nAll runs complete. Final Summary:")
    logging.info(" alpha    |   beta    |   final_maxE       |  last_max_amp ")
    logging.info("-----------------------------------------------------------")
    for (a, b, fm, la) in results:
        logging.info(f" {a:8.1e} | {b:8.1e} | {fm:15.3e} | {la:15.3e}")

    total_elapsed = global_end - global_start
    logging.info(f"\nTotal run time: {total_elapsed:.2f} s.")
    logging.info(f"Saved runs in '{base_params.output_dir}' if amplitude changed by more than 1%.\n")


# --------------------------------------------------------------------
# 9) Entry Point
# --------------------------------------------------------------------
if __name__ == "__main__":
    run_3d_extended_realtime_sweep()

#!/usr/bin/env python3
"""
Visualize Temporal Derivative of Safety Function

Computes h(t) and h(t+dt) where the human moves by 2 grids.
Plots dh/dt = (h(t+dt) - h(t)) / dt.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import os
import sys

# Add script directory to path to import modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from generate_occupancy_grid import create_occupancy_grid, generate_example_scene
from plot_boundary_vectors import (
    find_boundary, 
    compute_gradient_from_human, 
    solve_laplace_guidance,
    apply_social_tangent,
    solve_poisson_safety,
    CELL_SIZE, GRID_SIZE, DH0_HUMAN, DH0_OBSTACLE
)

def compute_h_for_human_pos(obstacles, human_rect):
    """Compute safety function h for a given human position."""
    # Create grid
    grid = create_occupancy_grid(GRID_SIZE, obstacles, human_rect)
    
    # Identify boundaries
    bound, obstacle_boundary, human_boundary, human_overlap = find_boundary(grid)
    all_human_boundary = human_boundary + human_overlap
    
    # Initialize guidance field
    guidance_x = np.zeros(grid.shape)
    guidance_y = np.zeros(grid.shape)
    
    # Set boundary gradients for obstacles
    for (i, j) in obstacle_boundary:
        guidance_x[i, j] = 0 # Will be solved by Laplace, just need non-zero bound? 
                             # Wait, solve_laplace_guidance needs initial boundary values.
                             # In plot_boundary_vectors, we use compute_boundary_gradient
                             # But actually solve_laplace_guidance handles boundary conditions internally 
                             # based on `all_boundary` list?
                             # Let's check solve_laplace_guidance implementation.
        pass

    # Actually plot_boundary_vectors.plot_guidance_field does this:
    # 1. Sets initial boundary values in guidance_x/y
    # 2. Calls solve_laplace_guidance
    
    # Replicating logic from plot_boundary_vectors.plot_guidance_field:
    
    # Initialize with border gradients
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            if i == 0: guidance_x[i, j] = 1.0
            if i == rows - 1: guidance_x[i, j] = -1.0
            if j == 0: guidance_y[i, j] = 1.0
            if j == cols - 1: guidance_y[i, j] = -1.0

    # Obstacle boundary gradients
    for (i, j) in obstacle_boundary:
        # Simple gradient away from obstacle
        gx, gy = 0.0, 0.0
        # Check neighbors to find free space direction
        # ... logic from compute_boundary_gradient ...
        # For simplicity let's use the helper if available, or just re-implement simple one
        # Ideally import compute_boundary_gradient but it's not exported cleanly?
        # Let's assume obstacle normals are handled or we need to compute them.
        
        # Actually solve_laplace_guidance takes guidance_x/y as INITIAL GUESS / BCs.
        # It fixes values at `all_boundary`.
        # So we MUST set guidance_x/y at boundary cells correctly.
        
        # Let's import compute_boundary_gradient if possible, or copy it.
        # It's defined in plot_boundary_vectors.py
        pass

    # We need compute_boundary_gradient to set BCs
    from plot_boundary_vectors import compute_boundary_gradient
    
    for (i, j) in obstacle_boundary:
        gx, gy = compute_boundary_gradient(i, j, grid, source_type=1)
        guidance_x[i, j] = gx * DH0_OBSTACLE
        guidance_y[i, j] = gy * DH0_OBSTACLE
        
    for (i, j) in all_human_boundary:
        # compute_gradient_from_human assumes we know human_rect or can deduce it?
        # It uses `grid` to find '2' cells.
        gx, gy = compute_gradient_from_human(i, j, grid)
        guidance_x[i, j] = gx * DH0_HUMAN
        guidance_y[i, j] = gy * DH0_HUMAN
        
    all_boundary = obstacle_boundary + all_human_boundary
    
    # Solve Laplace
    gx_solved, gy_solved = solve_laplace_guidance(grid, guidance_x, guidance_y, all_boundary)
    
    # Apply Social Tangent
    gx_biased, gy_biased = apply_social_tangent(gx_solved, gy_solved, grid, all_human_boundary)
    
    # Solve Poisson
    h = solve_poisson_safety(grid, gx_biased, gy_biased, all_boundary)
    
    return h, grid

def plot_temporal_derivative():
    print("Computing temporal derivative...")
    
    # Time t=0
    obstacles, human_rect_t0 = generate_example_scene()
    print(f"Human t0: {human_rect_t0}")
    h0, grid0 = compute_h_for_human_pos(obstacles, human_rect_t0)
    
    # Time t=1 (Human moves 4 grids DOWN => y - 4)
    # human_rect is (x, y, w, h)
    x, y, w, h_dim = human_rect_t0
    human_rect_t1 = (x, y - 4, w, h_dim)
    print(f"Human t1: {human_rect_t1}")
    h1, grid1 = compute_h_for_human_pos(obstacles, human_rect_t1)
    
    # Compute derivative
    # dh/dt approx h1 - h0
    dh_dt = h1 - h0
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Use distinct grayscale colormap centered at 0
    # 0 = Mid-gray
    # Positive = White (Increasing safety)
    # Negative = Black (Decreasing safety)
    
    # Find scale
    max_val = np.max(np.abs(dh_dt))
    print(f"Max abs derivative: {max_val}")
    
    # Create distinct custom grayscale map
    # 0.0 -> Black
    # 0.5 -> Gray
    # 1.0 -> White
    # We want -max -> Black, 0 -> Gray, +max -> White
    
    im = ax.imshow(dh_dt, cmap='gray', origin='lower', vmin=-max_val, vmax=max_val,
                   extent=[0, GRID_SIZE*CELL_SIZE, 0, GRID_SIZE*CELL_SIZE])
    
    # Overlay obstacles from t0 (static)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid0[i, j] == 1: # Obstacle
                rect = patches.Rectangle(
                    (j * CELL_SIZE, i * CELL_SIZE),
                    CELL_SIZE, CELL_SIZE,
                    linewidth=0, facecolor='black', alpha=0.3
                )
                ax.add_patch(rect)
                
    # Draw outline of human at t0 and t1?
    # t0 human (blue dash)
    x0, y0, w0, h0_dim = human_rect_t0
    rect0 = patches.Rectangle(
        (x0 * CELL_SIZE, y0 * CELL_SIZE),
        w0 * CELL_SIZE, h0_dim * CELL_SIZE,
        linewidth=2, edgecolor='blue', linestyle='--', facecolor='none'
    )
    ax.add_patch(rect0)
    
    # t1 human (red solid)
    x1, y1, w1, h1_dim = human_rect_t1
    rect1 = patches.Rectangle(
        (x1 * CELL_SIZE, y1 * CELL_SIZE),
        w1 * CELL_SIZE, h1_dim * CELL_SIZE,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect1)
    
    # ax.legend()
    # ax.set_title("Temporal Derivative of Safety Function (dh/dt)\nHuman moves 2 grids DOWN")
    # plt.colorbar(im, label='dh/dt')
    ax.axis('off')
    
    output_path = os.path.join(SCRIPT_DIR, 'temporal_derivative.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == '__main__':
    plot_temporal_derivative()

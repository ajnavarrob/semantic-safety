#!/usr/bin/env python3
"""
Class-Aware Boundary Vectors Visualization

Shows boundary gradient vectors (dh0) with different magnitudes:
- Human boundary: dh0 = 1.5 (larger arrows)
- Obstacle boundary: dh0 = 0.5 (smaller arrows)

Only shows boundary layer vectors, not the full guidance field.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from collections import deque
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Import the occupancy grid generator
from generate_occupancy_grid import (
    generate_example_scene, 
    create_occupancy_grid,
    GRID_SIZE,
    CELL_SIZE,
    COLOR_FREE,
    COLOR_OBSTACLE,
    COLOR_HUMAN
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Boundary gradient magnitudes
DH0_HUMAN = 1.5
DH0_OBSTACLE = 0.5

# Visualization parameters
ARROW_SCALE = 12
ARROW_WIDTH = 0.008

# Social bias parameters
SOCIAL_TANGENT_BIAS = 1.5
SOCIAL_TANGENT_LAYERS = 4


def find_boundary(grid):
    """
    Find boundary cells (free cells adjacent to occupied cells).
    
    Args:
        grid: occupancy grid where 0=free, 1=obstacle, 2=human
    
    Returns:
        bound: array where -1=obstacle, -2=human, 0=boundary, 1=free
        obstacle_boundary: list of (i, j) obstacle boundary cells
        human_boundary: list of (i, j) human boundary cells (not overlapping with obstacles)
        human_overlap: list of (i, j) human boundary cells that are near obstacles (render as dots)
    """
    rows, cols = grid.shape
    
    # Create bound array: -1=obstacle, -2=human, 1=free
    bound = np.ones((rows, cols), dtype=np.float32)
    bound[grid == 1] = -1.0  # obstacle
    bound[grid == 2] = -2.0  # human (different marker)
    
    obstacle_boundary = []
    human_boundary = []
    human_overlap = []  # Human cells that also touch obstacles - will be dots
    
    # Find boundary cells (free cells adjacent to occupied)
    for i in range(rows):
        for j in range(cols):
            if bound[i, j] == 1.0:  # free cell
                # Check 8-connected neighbors
                is_obstacle_adjacent = False
                is_human_adjacent = False
                
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            if bound[ni, nj] == -1.0:
                                is_obstacle_adjacent = True
                            elif bound[ni, nj] == -2.0:
                                is_human_adjacent = True
                
                if is_obstacle_adjacent:
                    obstacle_boundary.append((i, j))
                    # If also adjacent to human, mark for dot rendering
                    if is_human_adjacent:
                        human_overlap.append((i, j))
                elif is_human_adjacent:
                    # Check if this human boundary cell is close to any obstacle (within 2 cells)
                    near_obstacle = False
                    for di in range(-2, 3):
                        for dj in range(-2, 3):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < rows and 0 <= nj < cols:
                                if grid[ni, nj] == 1:  # obstacle
                                    near_obstacle = True
                                    break
                        if near_obstacle:
                            break
                    
                    if near_obstacle:
                        human_overlap.append((i, j))
                    else:
                        human_boundary.append((i, j))
    
    return bound, obstacle_boundary, human_boundary, human_overlap


def compute_boundary_gradient(i, j, grid, source_type):
    """
    Compute gradient at a boundary cell using simple 3x3 kernel.
    Returns normalized (gx, gy) pointing away from obstacle/human.
    
    Args:
        i, j: cell coordinates
        grid: occupancy grid
        source_type: 1 for obstacle, 2 for human
    """
    rows, cols = grid.shape
    gx, gy = 0.0, 0.0
    
    # Simple approach: find direction away from nearest occupied cells
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                if grid[ni, nj] == source_type:
                    # Point away from this occupied cell
                    gx -= di
                    gy -= dj
    
    # Normalize
    mag = np.sqrt(gx**2 + gy**2)
    if mag > 0:
        gx /= mag
        gy /= mag
    
    return gx, gy


def compute_social_layers(grid, human_boundary, num_layers):
    """
    Compute layers outward from human boundary using BFS.
    Returns dict mapping (i,j) to layer number (1-N for outer layers).
    Layer 0 cells are the boundary cells themselves.
    """
    rows, cols = grid.shape
    layer_map = {}
    queue = deque()
    
    # Seed with cells adjacent to human boundary (layer 1 start points)
    human_boundary_set = set(human_boundary)
    for (i, j) in human_boundary:
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                if grid[ni, nj] == 0 and (ni, nj) not in human_boundary_set:
                    if (ni, nj) not in layer_map:
                        # Check not too close to obstacle
                        near_obstacle = False
                        for di2 in range(-1, 2):
                            for dj2 in range(-1, 2):
                                ni2, nj2 = ni + di2, nj + dj2
                                if 0 <= ni2 < rows and 0 <= nj2 < cols:
                                    if grid[ni2, nj2] == 1:
                                        near_obstacle = True
                                        break
                            if near_obstacle:
                                break
                        if not near_obstacle:
                            layer_map[(ni, nj)] = 1
                            queue.append((ni, nj))
    
    # BFS to expand outward
    while queue:
        ci, cj = queue.popleft()
        cur_layer = layer_map[(ci, cj)]
        
        if cur_layer >= num_layers:
            continue
        
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = ci + di, cj + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                if (ni, nj) not in layer_map and (ni, nj) not in human_boundary_set and grid[ni, nj] == 0:
                    near_obstacle = False
                    for di2 in range(-1, 2):
                        for dj2 in range(-1, 2):
                            ni2, nj2 = ni + di2, nj + dj2
                            if 0 <= ni2 < rows and 0 <= nj2 < cols:
                                if grid[ni2, nj2] == 1:
                                    near_obstacle = True
                                    break
                        if near_obstacle:
                            break
                    if not near_obstacle:
                        layer_map[(ni, nj)] = cur_layer + 1
                        queue.append((ni, nj))
    
    return layer_map


def apply_tangent_bias(gx, gy, layer, num_layers, bias_strength):
    """
    Apply tangential bias to gradient for social navigation.
    CCW rotation for "pass on human's right" behavior.
    """
    # CCW tangent: (-gy, gx)
    tx, ty = -gy, gx
    
    # Decay with distance from boundary
    decay = 1.0 - (layer - 1) / num_layers
    sign = -1.0  # CCW bias
    
    biased_gx = gx + sign * bias_strength * decay * tx
    biased_gy = gy + sign * bias_strength * decay * ty
    
    mag = np.sqrt(biased_gx**2 + biased_gy**2)
    if mag > 0:
        biased_gx /= mag
        biased_gy /= mag
    
    return biased_gx, biased_gy


def compute_gradient_from_human(i, j, grid):
    """
    Compute gradient pointing away from the human center.
    Used for outer social layers that aren't adjacent to human.
    """
    human_cells = np.argwhere(grid == 2)
    if len(human_cells) == 0:
        return 0.0, 0.0
    
    # Human center
    center_i = human_cells[:, 0].mean()
    center_j = human_cells[:, 1].mean()
    
    # Direction away from center
    gx = i - center_i
    gy = j - center_j
    
    mag = np.sqrt(gx**2 + gy**2)
    if mag > 0:
        gx /= mag
        gy /= mag
    
    return gx, gy


def plot_boundary_vectors(grid, save_path=None):
    """
    Plot occupancy grid with class-aware boundary vectors.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor(COLOR_FREE)
    
    grid_size = grid.shape[0]
    
    # Draw obstacle cells
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i, j] == 1:  # obstacle
                rect = patches.Rectangle(
                    (j * CELL_SIZE, i * CELL_SIZE),
                    CELL_SIZE, CELL_SIZE,
                    linewidth=0, edgecolor='none', facecolor=COLOR_OBSTACLE
                )
                ax.add_patch(rect)
    
    # Draw human outline
    human_cells = np.argwhere(grid == 2)
    if len(human_cells) > 0:
        min_i, min_j = human_cells.min(axis=0)
        max_i, max_j = human_cells.max(axis=0)
        human_width = (max_j - min_j + 1) * CELL_SIZE
        human_height = (max_i - min_i + 1) * CELL_SIZE
        human_outline = patches.Rectangle(
            (min_j * CELL_SIZE, min_i * CELL_SIZE),
            human_width, human_height,
            linewidth=2.5, edgecolor=COLOR_HUMAN, facecolor='none'
        )
        ax.add_patch(human_outline)
    
    # Find boundaries
    bound, obstacle_boundary, human_boundary, human_overlap = find_boundary(grid)
    
    # Convert human_overlap to set for fast lookup
    human_overlap_set = set(human_overlap)
    
    # Find human bounding box for detecting left-side arrows
    human_cells = np.argwhere(grid == 2)
    human_min_j = human_cells[:, 1].min() if len(human_cells) > 0 else 0
    
    # Draw obstacle boundary vectors (dh0 = 0.5)
    # Skip cells that are in human_overlap (will be dots instead)
    for (i, j) in obstacle_boundary:
        if (i, j) in human_overlap_set:
            continue  # Skip - will draw orange dot instead
        gx, gy = compute_boundary_gradient(i, j, grid, source_type=1)
        if gx == 0 and gy == 0:
            continue
        # Scale by dh0
        arrow_len = DH0_OBSTACLE * 0.8
        ax.quiver(
            (j + 0.5) * CELL_SIZE, (i + 0.5) * CELL_SIZE,
            gy * arrow_len, gx * arrow_len,
            color='#2196F3',  # Blue for obstacles
            scale=ARROW_SCALE, width=ARROW_WIDTH,
            headwidth=4, headlength=5,
            zorder=10
        )
    
    # Draw orange dots for overlapping human/obstacle boundary cells
    for (i, j) in human_overlap:
        ax.plot(
            (j + 0.5) * CELL_SIZE, (i + 0.5) * CELL_SIZE,
            'o', color='#FF5722', markersize=6, zorder=12
        )
    
    # Draw human boundary vectors (dh0 = 1.5)
    for (i, j) in human_boundary:
        gx, gy = compute_boundary_gradient(i, j, grid, source_type=2)
        if gx == 0 and gy == 0:
            continue
        
        # Reduce arrow length on left side of human (where gy points left)
        arrow_len = DH0_HUMAN * 0.8
        if j < human_min_j + 3 and gy < 0:
            arrow_len = DH0_HUMAN * 0.4  # Shorter arrows on left side
        
        ax.quiver(
            (j + 0.5) * CELL_SIZE, (i + 0.5) * CELL_SIZE,
            gy * arrow_len, gx * arrow_len,
            color='#FF5722',  # Orange-red for human
            scale=ARROW_SCALE, width=ARROW_WIDTH,
            headwidth=4, headlength=5,
            zorder=11
        )
    
    # Draw grid lines
    for i in range(grid_size + 1):
        ax.axhline(y=i * CELL_SIZE, color='#CCCCCC', linewidth=0.3, alpha=0.5)
        ax.axvline(x=i * CELL_SIZE, color='#CCCCCC', linewidth=0.3, alpha=0.5)
    
    # Border
    border = patches.Rectangle(
        (0, 0), grid_size * CELL_SIZE, grid_size * CELL_SIZE,
        linewidth=2, edgecolor='black', facecolor='none'
    )
    ax.add_patch(border)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='>', color='#2196F3', linestyle='None',
               markersize=12, label=f'Obstacle dh0 = {DH0_OBSTACLE}'),
        Line2D([0], [0], marker='>', color='#FF5722', linestyle='None',
               markersize=12, label=f'Human dh0 = {DH0_HUMAN}'),
    ]
    # ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    ax.set_xlim(0, grid_size * CELL_SIZE)
    ax.set_ylim(0, grid_size * CELL_SIZE)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # plt.title('Class-Aware Boundary Vectors', fontsize=14, fontweight='bold', pad=10)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_social_bias_layers(grid, save_path=None):
    """
    Plot social bias layers around human, starting at boundary endpoints.
    Only shows bias layers at top, bottom, and lower right (not left side near obstacles).
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor(COLOR_FREE)
    
    grid_size = grid.shape[0]
    
    # Draw obstacle cells
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i, j] == 1:
                rect = patches.Rectangle(
                    (j * CELL_SIZE, i * CELL_SIZE),
                    CELL_SIZE, CELL_SIZE,
                    linewidth=0, edgecolor='none', facecolor=COLOR_OBSTACLE
                )
                ax.add_patch(rect)
    
    # Draw human outline
    human_cells = np.argwhere(grid == 2)
    if len(human_cells) > 0:
        min_i, min_j = human_cells.min(axis=0)
        max_i, max_j = human_cells.max(axis=0)
        human_width = (max_j - min_j + 1) * CELL_SIZE
        human_height = (max_i - min_i + 1) * CELL_SIZE
        human_outline = patches.Rectangle(
            (min_j * CELL_SIZE, min_i * CELL_SIZE),
            human_width, human_height,
            linewidth=2.5, edgecolor=COLOR_HUMAN, facecolor='none'
        )
        ax.add_patch(human_outline)
        
        # Get human bounding box for filtering sides
        human_center_j = (min_j + max_j) / 2
    
    # Find human boundaries (not near obstacles)
    bound, obstacle_boundary, human_boundary, human_overlap = find_boundary(grid)
    
    # Get human bounding box for side detection
    human_min_j = human_cells[:, 1].min() if len(human_cells) > 0 else 0
    human_max_j = human_cells[:, 1].max() if len(human_cells) > 0 else grid_size
    
    # Draw human boundary vectors (the starting point for social layers)
    for (i, j) in human_boundary:
        gx, gy = compute_boundary_gradient(i, j, grid, source_type=2)
        if gx == 0 and gy == 0:
            continue
        
        arrow_len = DH0_HUMAN * 0.8
        if j < human_min_j + 3 and gy < 0:
            arrow_len = DH0_HUMAN * 0.4
        
        ax.quiver(
            (j + 0.5) * CELL_SIZE, (i + 0.5) * CELL_SIZE,
            gy * arrow_len, gx * arrow_len,
            color='#FF5722',
            scale=ARROW_SCALE, width=ARROW_WIDTH,
            headwidth=4, headlength=5,
            zorder=11
        )
    
    # Draw orange dots for overlap regions
    for (i, j) in human_overlap:
        ax.plot(
            (j + 0.5) * CELL_SIZE, (i + 0.5) * CELL_SIZE,
            'o', color='#FF5722', markersize=6, zorder=12
        )
    
    # Compute social layers
    social_layers = compute_social_layers(grid, human_boundary, SOCIAL_TANGENT_LAYERS)
    
    # Layer colors: green gradient
    layer_colors = ['#4CAF50', '#66BB6A', '#81C784', '#A5D6A7']
    
    # Draw social bias arrows only at top, bottom, and right sides
    for (i, j), layer in social_layers.items():
        # Skip left side (where cell is left of human)
        if j < human_min_j:
            continue
        
        # Use direction from human center for consistent gradients
        gx, gy = compute_gradient_from_human(i, j, grid)
        if gx == 0 and gy == 0:
            continue
        
        # Apply tangent bias
        biased_gx, biased_gy = apply_tangent_bias(
            gx, gy, layer, SOCIAL_TANGENT_LAYERS, SOCIAL_TANGENT_BIAS
        )
        
        color = layer_colors[min(layer - 1, len(layer_colors) - 1)]
        arrow_len = DH0_HUMAN * 0.5
        
        ax.quiver(
            (j + 0.5) * CELL_SIZE, (i + 0.5) * CELL_SIZE,
            biased_gy * arrow_len, biased_gx * arrow_len,
            color=color,
            scale=ARROW_SCALE, width=ARROW_WIDTH * 0.8,
            headwidth=4, headlength=5,
            zorder=8 + layer
        )
    
    # Draw grid lines
    for i in range(grid_size + 1):
        ax.axhline(y=i * CELL_SIZE, color='#CCCCCC', linewidth=0.3, alpha=0.5)
        ax.axvline(x=i * CELL_SIZE, color='#CCCCCC', linewidth=0.3, alpha=0.5)
    
    # Border
    border = patches.Rectangle(
        (0, 0), grid_size * CELL_SIZE, grid_size * CELL_SIZE,
        linewidth=2, edgecolor='black', facecolor='none'
    )
    ax.add_patch(border)
    
    ax.set_xlim(0, grid_size * CELL_SIZE)
    ax.set_ylim(0, grid_size * CELL_SIZE)
    ax.set_aspect('equal')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {save_path}")
    
    plt.close()


def solve_laplace_guidance(grid, guidance_x, guidance_y, boundary_cells):
    """
    Solve Laplace equation to smooth guidance field throughout free space.
    Boundary cells are fixed Dirichlet conditions.
    """
    rows, cols = grid.shape
    n_cells = rows * cols
    
    # Create system Ax = b using sparse matrix
    A = lil_matrix((n_cells, n_cells))
    bx = np.zeros(n_cells)
    by = np.zeros(n_cells)
    
    boundary_set = set(boundary_cells)
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            
            if grid[i, j] != 0:  # Occupied cell
                A[idx, idx] = 1.0
                bx[idx] = 0.0
                by[idx] = 0.0
            elif (i, j) in boundary_set:  # Boundary cell - Dirichlet BC
                A[idx, idx] = 1.0
                bx[idx] = guidance_x[i, j]
                by[idx] = guidance_y[i, j]
            else:  # Free space - Laplace equation
                A[idx, idx] = 4.0
                neighbors = []
                if i > 0:
                    neighbors.append((i-1, j))
                if i < rows - 1:
                    neighbors.append((i+1, j))
                if j > 0:
                    neighbors.append((i, j-1))
                if j < cols - 1:
                    neighbors.append((i, j+1))
                
                for ni, nj in neighbors:
                    nidx = ni * cols + nj
                    A[idx, nidx] = -1.0
                
                bx[idx] = 0.0
                by[idx] = 0.0
    
    # Solve
    A_csr = A.tocsr()
    gx_flat = spsolve(A_csr, bx)
    gy_flat = spsolve(A_csr, by)
    
    return gx_flat.reshape((rows, cols)), gy_flat.reshape((rows, cols))


def apply_social_tangent(guidance_x, guidance_y, grid, human_boundary):
    """
    Apply tangential bias to guidance field in layers around human.
    Layer 0 (boundary): pure normal for CBF guarantee
    Layers 1-N: tangentially biased for social navigation
    """
    gx = guidance_x.copy()
    gy = guidance_y.copy()
    rows, cols = grid.shape
    
    if len(human_boundary) == 0:
        return gx, gy
    
    # BFS to find distance from human boundary
    dist_map = np.full((rows, cols), np.inf)
    queue = deque()
    
    for (i, j) in human_boundary:
        dist_map[i, j] = 0
        queue.append((i, j))
    
    while queue:
        ci, cj = queue.popleft()
        cur_dist = dist_map[ci, cj]
        
        if cur_dist >= SOCIAL_TANGENT_LAYERS:
            continue
        
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = ci + di, cj + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                if dist_map[ni, nj] == np.inf and grid[ni, nj] == 0:
                    dist_map[ni, nj] = cur_dist + 1
                    queue.append((ni, nj))
    
    # Apply tangent bias to layers 1-N (skip layer 0)
    for i in range(rows):
        for j in range(cols):
            dist = dist_map[i, j]
            if 1 <= dist <= SOCIAL_TANGENT_LAYERS:
                mag = np.sqrt(gx[i, j]**2 + gy[i, j]**2)
                if mag < 0.01:
                    continue
                
                gx_n = gx[i, j] / mag
                gy_n = gy[i, j] / mag
                
                # CCW tangent: (-gy, gx)
                decay = 1.0 - (dist - 1) / SOCIAL_TANGENT_LAYERS
                sign = -1.0  # CCW (counterclockwise)
                
                biased_gx = gx_n + sign * SOCIAL_TANGENT_BIAS * decay * (-gy_n)
                biased_gy = gy_n + sign * SOCIAL_TANGENT_BIAS * decay * gx_n
                
                V = np.sqrt(biased_gx**2 + biased_gy**2)
                if V > 0:
                    gx[i, j] = biased_gx / V * mag
                    gy[i, j] = biased_gy / V * mag
    
    return gx, gy


def solve_poisson_safety(grid, guidance_x, guidance_y, boundary_cells):
    """
    Solve Poisson equation nabla^2 h = div(g) for safety function h.
    Boundary conditions: h = 0 at boundaries.
    """
    rows, cols = grid.shape
    n_cells = rows * cols
    
    # Compute divergence of guidance field
    # Use central differences
    dgx_di = (np.roll(guidance_x, -1, axis=0) - np.roll(guidance_x, 1, axis=0)) / (2.0 * CELL_SIZE)
    dgy_dj = (np.roll(guidance_y, -1, axis=1) - np.roll(guidance_y, 1, axis=1)) / (2.0 * CELL_SIZE)
    div = dgx_di + dgy_dj
    
    # Forcing function f * DS^2
    # nabla^2 h = div(g)
    # Discrete: 4h - sum(h_neigh) = -div(g) * DS^2
    force = div * CELL_SIZE * CELL_SIZE
    
    # Create system Ax = b
    A = lil_matrix((n_cells, n_cells))
    b = np.zeros(n_cells)
    
    boundary_set = set(boundary_cells)
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            
            if grid[i, j] != 0:  # Occupied/Human cells
                # Inside obstacles, valid h doesn't matter much for safety, 
                # strictly it should be negative distance, but we focus on free space.
                # Let's just fix it to 0 or solve continuity.
                # Using 0 for simplicity or same as boundary.
                A[idx, idx] = 1.0
                b[idx] = 0.0
            elif (i, j) in boundary_set:  # Boundary cells
                A[idx, idx] = 1.0
                b[idx] = 0.0  # Dirichlet BC h=0
            else:  # Free space
                A[idx, idx] = 4.0
                neighbors = []
                if i > 0: neighbors.append((i-1, j))
                if i < rows - 1: neighbors.append((i+1, j))
                if j > 0: neighbors.append((i, j-1))
                if j < cols - 1: neighbors.append((i, j+1))
                
                for ni, nj in neighbors:
                    nidx = ni * cols + nj
                    A[idx, nidx] = -1.0
                
                # Equation: 4h - sum(h) = -force
                b[idx] = -force[i, j]
    
    # Solve
    A_csr = A.tocsr()
    h_flat = spsolve(A_csr, b)
    
    return h_flat.reshape((rows, cols))


def plot_safety_function(grid, h_values, save_path=None):
    """
    Plot safety function h as a heatmap.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    im = ax.imshow(h_values, cmap='viridis', origin='lower', 
                   extent=[0, grid.shape[1]*CELL_SIZE, 0, grid.shape[0]*CELL_SIZE])
    # plt.colorbar(im)
    
    # Overlay obstacles (semi-transparent)
    grid_size = grid.shape[0]
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i, j] == 1: # Obstacle
                rect = patches.Rectangle(
                    (j * CELL_SIZE, i * CELL_SIZE),
                    CELL_SIZE, CELL_SIZE,
                    linewidth=0, facecolor='black', alpha=0.5
                )
                ax.add_patch(rect)
    
    # Human outline
    human_cells = np.argwhere(grid == 2)
    if len(human_cells) > 0:
        min_i, min_j = human_cells.min(axis=0)
        max_i, max_j = human_cells.max(axis=0)
        human_width = (max_j - min_j + 1) * CELL_SIZE
        human_height = (max_i - min_i + 1) * CELL_SIZE
        human_outline = patches.Rectangle(
            (min_j * CELL_SIZE, min_i * CELL_SIZE),
            human_width, human_height,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(human_outline)

    # Plot guidance field arrows over it (subsampled)
    # Using the guidance field associated with h would be cool, but we usually
    # plot the one we used to generate it.
    # For now just the heatmap is enough as per request "plot the poisson function"
    
    # ax.set_title('Poisson Safety Function h')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_guidance_field(grid, save_path=None):
    """
    Plot continuous guidance field solved via Laplace with social biasing.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor(COLOR_FREE)
    
    grid_size = grid.shape[0]
    
    # Draw obstacle cells
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i, j] == 1:
                rect = patches.Rectangle(
                    (j * CELL_SIZE, i * CELL_SIZE),
                    CELL_SIZE, CELL_SIZE,
                    linewidth=0, edgecolor='none', facecolor=COLOR_OBSTACLE
                )
                ax.add_patch(rect)
    
    # Draw human outline
    human_cells = np.argwhere(grid == 2)
    if len(human_cells) > 0:
        min_i, min_j = human_cells.min(axis=0)
        max_i, max_j = human_cells.max(axis=0)
        human_width = (max_j - min_j + 1) * CELL_SIZE
        human_height = (max_i - min_i + 1) * CELL_SIZE
        human_outline = patches.Rectangle(
            (min_j * CELL_SIZE, min_i * CELL_SIZE),
            human_width, human_height,
            linewidth=2.5, edgecolor=COLOR_HUMAN, facecolor='none'
        )
        ax.add_patch(human_outline)
    
    # Find all boundary cells (obstacle + human)
    bound, obstacle_boundary, human_boundary, human_overlap = find_boundary(grid)
    
    # Initialize guidance arrays
    guidance_x = np.zeros((grid_size, grid_size))
    guidance_y = np.zeros((grid_size, grid_size))
    
    # Set boundary conditions for obstacles
    for (i, j) in obstacle_boundary:
        gx, gy = compute_boundary_gradient(i, j, grid, source_type=1)
        guidance_x[i, j] = gx * DH0_OBSTACLE
        guidance_y[i, j] = gy * DH0_OBSTACLE
    
    # Set boundary conditions for human (including overlap as dots -> still set BC)
    all_human_boundary = human_boundary + human_overlap
    for (i, j) in all_human_boundary:
        gx, gy = compute_boundary_gradient(i, j, grid, source_type=2)
        if gx == 0 and gy == 0:
            gx, gy = compute_gradient_from_human(i, j, grid)
        guidance_x[i, j] = gx * DH0_HUMAN
        guidance_y[i, j] = gy * DH0_HUMAN
    
    # Combine all boundary cells
    all_boundary = obstacle_boundary + all_human_boundary
    
    # Solve Laplace equation
    gx_solved, gy_solved = solve_laplace_guidance(grid, guidance_x, guidance_y, all_boundary)
    
    # Apply social tangent bias
    gx_biased, gy_biased = apply_social_tangent(gx_solved, gy_solved, grid, all_human_boundary)
    
    # Draw obstacle boundary arrows (blue)
    for (i, j) in obstacle_boundary:
        gx = gx_solved[i, j]
        gy = gy_solved[i, j]
        mag = np.sqrt(gx**2 + gy**2)
        if mag > 0.01:
            arrow_len = 0.25
            ax.quiver(
                (j + 0.5) * CELL_SIZE, (i + 0.5) * CELL_SIZE,
                gy / mag * arrow_len, gx / mag * arrow_len,
                color='#2196F3', alpha=1.0,
                scale=ARROW_SCALE * 1.2, width=ARROW_WIDTH * 0.9,
                headwidth=3, headlength=4,
                zorder=9
            )
    
    # Draw layer 0 normal arrows around human (all cells, shorter near obstacles)
    for (i, j) in all_human_boundary:
        # Check if near an obstacle
        near_obstacle = False
        for di in range(-3, 4):
            for dj in range(-3, 4):
                ni, nj = i + di, j + dj
                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                    if grid[ni, nj] == 1:
                        near_obstacle = True
                        break
            if near_obstacle:
                break
        
        gx = gx_solved[i, j]
        gy = gy_solved[i, j]
        mag = np.sqrt(gx**2 + gy**2)
        if mag > 0.01:
            # Use very short arrows near obstacles, normal short arrows elsewhere
            arrow_len = 0.15 if near_obstacle else 0.3
            ax.quiver(
                (j + 0.5) * CELL_SIZE, (i + 0.5) * CELL_SIZE,
                gy / mag * arrow_len, gx / mag * arrow_len,
                color='#FF5722', alpha=1.0,
                scale=ARROW_SCALE * 1.2, width=ARROW_WIDTH * 0.9,
                headwidth=3, headlength=4,
                zorder=10
            )
    
    # Draw continuous quiver field (subsample for visibility)
    step = 2
    all_boundary_set = set(all_boundary)
    for i in range(0, grid_size, step):
        for j in range(0, grid_size, step):
            if grid[i, j] != 0:  # Skip obstacles and human
                continue
            
            # Skip boundary cells (already drawn separately)
            if (i, j) in all_boundary_set:
                continue
            
            gx = gx_biased[i, j]
            gy = gy_biased[i, j]
            mag = np.sqrt(gx**2 + gy**2)
            
            if mag < 0.01:
                continue
            
            # Color based on magnitude (blue=obstacle-like, orange=human-like)
            if mag < DH0_OBSTACLE + 0.2:
                color = '#2196F3'  # Blue
            elif mag < DH0_HUMAN - 0.2:
                color = '#4CAF50'  # Green
            else:
                color = '#FF5722'  # Orange
            
            arrow_len = mag * 0.6
            
            ax.quiver(
                (j + 0.5) * CELL_SIZE, (i + 0.5) * CELL_SIZE,
                gy * arrow_len, gx * arrow_len,
                color=color, alpha=1.0,
                scale=ARROW_SCALE * 1.2, width=ARROW_WIDTH * 0.9,
                headwidth=3, headlength=4,
                zorder=5
            )
    
    # Draw grid lines
    for i in range(grid_size + 1):
        ax.axhline(y=i * CELL_SIZE, color='#CCCCCC', linewidth=0.3, alpha=0.3)
        ax.axvline(x=i * CELL_SIZE, color='#CCCCCC', linewidth=0.3, alpha=0.3)
    
    # Border
    border = patches.Rectangle(
        (0, 0), grid_size * CELL_SIZE, grid_size * CELL_SIZE,
        linewidth=2, edgecolor='black', facecolor='none'
    )
    ax.add_patch(border)
    
    ax.set_xlim(0, grid_size * CELL_SIZE)
    ax.set_ylim(0, grid_size * CELL_SIZE)
    ax.set_aspect('equal')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {save_path}")
    
    plt.close()
    
    return gx_biased, gy_biased


def main():
    """Generate boundary vectors, social bias, and guidance field visualizations."""
    print("Generating visualizations...")
    
    # Get the example scene
    obstacles, human_rect = generate_example_scene()
    grid = create_occupancy_grid(GRID_SIZE, obstacles, human_rect)
    
    # Plot boundary vectors only
    output_path = os.path.join(SCRIPT_DIR, 'boundary_vectors.png')
    plot_boundary_vectors(grid, save_path=output_path)
    
    # Plot social bias layers (separate figure)
    social_output_path = os.path.join(SCRIPT_DIR, 'social_bias_layers.png')
    plot_social_bias_layers(grid, save_path=social_output_path)
    
    # Plot continuous guidance field with Laplace solve
    guidance_output_path = os.path.join(SCRIPT_DIR, 'guidance_field.png')
    gx_final, gy_final = plot_guidance_field(grid, save_path=guidance_output_path)
    
    # Solve Poisson for safety function
    bound, obstacle_boundary, human_boundary, human_overlap = find_boundary(grid)
    all_boundary = obstacle_boundary + human_boundary + human_overlap
    h_values = solve_poisson_safety(grid, gx_final, gy_final, all_boundary)
    
    # Plot safety function
    poisson_output_path = os.path.join(SCRIPT_DIR, 'poisson_safety_function.png')
    plot_safety_function(grid, h_values, save_path=poisson_output_path)
    
    print("Done!")


if __name__ == '__main__':
    main()



#!/usr/bin/env python3
"""
Social Biasing Layer Visualization

Shows the tangential bias layers around the human for social navigation.
- Layer 0 (boundary): Normal vectors only (CBF guarantee)
- Layers 1-N: Tangentially biased vectors (social navigation)
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import os

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

# Parameters
DH0_HUMAN = 1.5
DH0_OBSTACLE = 0.5
SOCIAL_TANGENT_BIAS = 1.5
SOCIAL_TANGENT_LAYERS = 4

ARROW_SCALE = 12
ARROW_WIDTH = 0.008


def find_human_boundary(grid):
    """Find boundary cells adjacent to human only."""
    rows, cols = grid.shape
    human_boundary = []
    
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 0:  # free cell
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            if grid[ni, nj] == 2:  # human
                                human_boundary.append((i, j))
                                break
                    else:
                        continue
                    break
    
    return human_boundary


def compute_social_layers(grid, human_boundary, num_layers):
    """
    Compute layers outward from human boundary using BFS.
    Returns dict mapping (i,j) to layer number (0 = boundary, 1-N = outer layers).
    """
    from collections import deque
    
    rows, cols = grid.shape
    layer_map = {}
    queue = deque()
    
    # Seed with boundary cells (layer 0)
    for (i, j) in human_boundary:
        layer_map[(i, j)] = 0
        queue.append((i, j))
    
    # BFS to find layers
    while queue:
        ci, cj = queue.popleft()
        cur_layer = layer_map[(ci, cj)]
        
        if cur_layer >= num_layers:
            continue
        
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = ci + di, cj + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                if (ni, nj) not in layer_map and grid[ni, nj] == 0:
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
                        layer_map[(ni, nj)] = cur_layer + 1
                        queue.append((ni, nj))
    
    return layer_map


def compute_gradient_toward_human(i, j, grid):
    """Compute normalized gradient pointing away from human."""
    rows, cols = grid.shape
    gx, gy = 0.0, 0.0
    
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                if grid[ni, nj] == 2:  # human
                    gx -= di
                    gy -= dj
    
    mag = np.sqrt(gx**2 + gy**2)
    if mag > 0:
        gx /= mag
        gy /= mag
    
    return gx, gy


def apply_tangent_bias(gx, gy, layer, num_layers, bias_strength):
    """
    Apply tangential bias to gradient.
    Layer 0: no bias (normal only)
    Layers 1-N: decaying tangent bias
    """
    if layer == 0:
        return gx, gy
    
    # CCW tangent: rotate 90° counterclockwise -> (-gy, gx)
    tx, ty = -gy, gx
    
    # Decay with distance
    decay = 1.0 - (layer - 1) / num_layers
    sign = -1.0  # CCW bias (robot passes on human's right)
    
    biased_gx = gx + sign * bias_strength * decay * tx
    biased_gy = gy + sign * bias_strength * decay * ty
    
    # Normalize
    mag = np.sqrt(biased_gx**2 + biased_gy**2)
    if mag > 0:
        biased_gx /= mag
        biased_gy /= mag
    
    return biased_gx, biased_gy


def plot_social_bias_layers(grid, save_path=None):
    """Plot occupancy grid with social biasing layers around human."""
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
    
    # Find human boundary and compute layers
    human_boundary = find_human_boundary(grid)
    layer_map = compute_social_layers(grid, human_boundary, SOCIAL_TANGENT_LAYERS)
    
    # Layer colors: boundary (cyan) -> layers (gold -> orange -> light)
    layer_colors = ['#00CED1', '#FFD700', '#FFA500', '#FF8C00', '#FF6347']
    
    # Draw arrows for each layer
    for (i, j), layer in layer_map.items():
        gx, gy = compute_gradient_toward_human(i, j, grid)
        if gx == 0 and gy == 0:
            continue
        
        # Apply tangent bias for layers > 0
        biased_gx, biased_gy = apply_tangent_bias(
            gx, gy, layer, SOCIAL_TANGENT_LAYERS, SOCIAL_TANGENT_BIAS
        )
        
        color = layer_colors[min(layer, len(layer_colors) - 1)]
        arrow_len = DH0_HUMAN * 0.6
        
        ax.quiver(
            (j + 0.5) * CELL_SIZE, (i + 0.5) * CELL_SIZE,
            biased_gy * arrow_len, biased_gx * arrow_len,
            color=color,
            scale=ARROW_SCALE, width=ARROW_WIDTH,
            headwidth=4, headlength=5,
            zorder=10 + layer
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


def main():
    """Generate social bias layers visualization."""
    print("Generating social bias layers visualization...")
    
    obstacles, human_rect = generate_example_scene()
    grid = create_occupancy_grid(GRID_SIZE, obstacles, human_rect)
    
    output_path = os.path.join(SCRIPT_DIR, 'social_bias_layers.png')
    plot_social_bias_layers(grid, save_path=output_path)
    
    print("Done!")


if __name__ == '__main__':
    main()

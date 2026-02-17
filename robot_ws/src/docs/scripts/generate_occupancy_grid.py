#!/usr/bin/env python3
"""
Occupancy Grid Generator with Rectangle Obstacles

Generates a 2D occupancy grid visualization with rectangular obstacles,
including a designated 'human' rectangle with a distinct appearance.

Output: An image showing the grid with:
- Free space (light gray/white)
- Obstacles (dark gray)
- Human (red/distinct color)
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless rendering

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Output directory (same as script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Grid parameters
GRID_SIZE = 50  # Grid dimensions (GRID_SIZE x GRID_SIZE cells)
CELL_SIZE = 1.0  # Size of each cell in arbitrary units

# Colors
COLOR_FREE = '#F5F5F5'      # Light gray for free space
COLOR_OBSTACLE = '#4A4A4A'  # Dark gray for obstacles
COLOR_HUMAN = '#E74C3C'     # Red for human
COLOR_GRID = '#CCCCCC'      # Grid line color


def create_occupancy_grid(grid_size, obstacles, human_rect=None):
    """
    Create an occupancy grid with rectangular obstacles.
    
    Args:
        grid_size: Size of the grid (grid_size x grid_size)
        obstacles: List of obstacle rectangles [(x, y, width, height), ...]
                   where (x, y) is the bottom-left corner
        human_rect: Optional human rectangle (x, y, width, height)
    
    Returns:
        grid: 2D numpy array where:
               0 = free space
               1 = obstacle
               2 = human
    """
    grid = np.zeros((grid_size, grid_size), dtype=np.int8)
    
    # Place obstacles
    for (x, y, w, h) in obstacles:
        x, y, w, h = int(x), int(y), int(w), int(h)
        x_end = min(x + w, grid_size)
        y_end = min(y + h, grid_size)
        grid[y:y_end, x:x_end] = 1
    
    # Place human (overwrites obstacle if overlapping)
    if human_rect is not None:
        x, y, w, h = human_rect
        x, y, w, h = int(x), int(y), int(w), int(h)
        x_end = min(x + w, grid_size)
        y_end = min(y + h, grid_size)
        grid[y:y_end, x:x_end] = 2
    
    return grid


def plot_occupancy_grid(grid, show_grid_lines=True, save_path=None):
    """
    Plot the occupancy grid with colored rectangles.
    
    Args:
        grid: 2D numpy array from create_occupancy_grid
        show_grid_lines: Whether to show grid lines
        save_path: Path to save the image (if None, displays interactively)
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    grid_size = grid.shape[0]
    
    # Fill background with free space color
    ax.set_facecolor(COLOR_FREE)
    
    # Draw cells
    for i in range(grid_size):
        for j in range(grid_size):
            cell_value = grid[i, j]
            
            if cell_value == 0:
                # Free space - already covered by background
                continue
            elif cell_value == 1:
                # Obstacle
                color = COLOR_OBSTACLE
                # Draw filled rectangle for obstacle
                rect = patches.Rectangle(
                    (j * CELL_SIZE, i * CELL_SIZE),
                    CELL_SIZE, CELL_SIZE,
                    linewidth=0,
                    edgecolor='none',
                    facecolor=color
                )
                ax.add_patch(rect)
            # Human cells (value 2) are drawn separately as outline
    
    # Draw grid lines
    if show_grid_lines:
        for i in range(grid_size + 1):
            ax.axhline(y=i * CELL_SIZE, color=COLOR_GRID, linewidth=0.3, alpha=0.5)
            ax.axvline(x=i * CELL_SIZE, color=COLOR_GRID, linewidth=0.3, alpha=0.5)
    
    # Add border
    border = patches.Rectangle(
        (0, 0), grid_size * CELL_SIZE, grid_size * CELL_SIZE,
        linewidth=2, edgecolor='black', facecolor='none'
    )
    ax.add_patch(border)
    
    # Draw human as outline (find bounding box of human cells)
    human_cells = np.argwhere(grid == 2)
    if len(human_cells) > 0:
        min_i, min_j = human_cells.min(axis=0)
        max_i, max_j = human_cells.max(axis=0)
        human_width = (max_j - min_j + 1) * CELL_SIZE
        human_height = (max_i - min_i + 1) * CELL_SIZE
        human_outline = patches.Rectangle(
            (min_j * CELL_SIZE, min_i * CELL_SIZE),
            human_width, human_height,
            linewidth=2.5,
            edgecolor=COLOR_HUMAN,
            facecolor='none'
        )
        ax.add_patch(human_outline)
    
    # Set axis properties
    ax.set_xlim(0, grid_size * CELL_SIZE)
    ax.set_ylim(0, grid_size * CELL_SIZE)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    # plt.title('Occupancy Grid with Obstacles and Human', fontsize=14, fontweight='bold', pad=10)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_example_scene():
    """
    Generate scene with disconnected walls:
    - Left wall (vertical, separate)
    - Right wall (vertical, separate)  
    - Bottom wall (horizontal, separate with gaps)
    - Upper-center vertical wall hanging down
    Human in the open passage area.
    """
    # Define rectangular obstacles (x, y, width, height)
    obstacles = [
        # Left wall (vertical, not touching corners)
        (0, 12, 8, 30),
        
        # Right wall (vertical, not touching corners)
        (42, 8, 8, 28),
        
        # Bottom wall (in the middle, with gaps on sides)
        (15, 0, 20, 8),
        
        # Upper-center vertical wall hanging down from top
        (27, 28, 6, 22),
    ]
    
    # Human rectangle in the open area (left of center wall)
    human_rect = (12, 22, 9, 16)
    
    return obstacles, human_rect


def generate_random_obstacles(grid_size, num_obstacles, min_size=3, max_size=10, seed=42):
    """
    Generate random rectangular obstacles.
    
    Args:
        grid_size: Size of the grid
        num_obstacles: Number of obstacles to generate
        min_size: Minimum obstacle dimension
        max_size: Maximum obstacle dimension
        seed: Random seed for reproducibility
    
    Returns:
        List of obstacle rectangles [(x, y, width, height), ...]
    """
    np.random.seed(seed)
    obstacles = []
    
    for _ in range(num_obstacles):
        w = np.random.randint(min_size, max_size + 1)
        h = np.random.randint(min_size, max_size + 1)
        x = np.random.randint(0, grid_size - w)
        y = np.random.randint(0, grid_size - h)
        obstacles.append((x, y, w, h))
    
    return obstacles


def main():
    """Main function to generate and save the occupancy grid."""
    
    # Example 1: Manually defined scene (similar to reference image)
    print("Generating example scene...")
    obstacles, human_rect = generate_example_scene()
    grid = create_occupancy_grid(GRID_SIZE, obstacles, human_rect)
    output_path = os.path.join(SCRIPT_DIR, 'occupancy_grid_example.png')
    plot_occupancy_grid(grid, show_grid_lines=True, save_path=output_path)
    
    # Example 2: Random obstacles
    print("Generating random obstacle scene...")
    random_obstacles = generate_random_obstacles(GRID_SIZE, num_obstacles=15, seed=123)
    human_rect_random = (20, 25, 4, 5)
    grid_random = create_occupancy_grid(GRID_SIZE, random_obstacles, human_rect_random)
    output_path_random = os.path.join(SCRIPT_DIR, 'occupancy_grid_random.png')
    plot_occupancy_grid(grid_random, show_grid_lines=True, save_path=output_path_random)
    
    # Example 3: Simple corridor scene
    print("Generating corridor scene...")
    corridor_obstacles = [
        # Top wall
        (0, 40, 50, 10),
        # Bottom wall
        (0, 0, 50, 10),
        # Center obstacles
        (10, 20, 8, 15),
        (30, 15, 8, 18),
    ]
    human_corridor = (20, 22, 4, 5)
    grid_corridor = create_occupancy_grid(GRID_SIZE, corridor_obstacles, human_corridor)
    output_path_corridor = os.path.join(SCRIPT_DIR, 'occupancy_grid_corridor.png')
    plot_occupancy_grid(grid_corridor, show_grid_lines=True, save_path=output_path_corridor)
    
    print("Done! Generated 3 occupancy grid images.")


if __name__ == '__main__':
    main()

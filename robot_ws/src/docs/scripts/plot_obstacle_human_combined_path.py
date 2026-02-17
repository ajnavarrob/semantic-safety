#!/usr/bin/env python3
"""
Visualization: Combined Obstacle + Human in Same Poisson Field

Shows obstacle on left, human on right in the SAME grid:
- Row 0: 2D Guidance Field + Safety Function (combined scene)
- Row 1: 3D Poisson Safety Function Surface (combined scene)
"""

import matplotlib
matplotlib.use('Agg')

# Set global font to Times New Roman
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman'] + matplotlib.rcParams['font.serif']

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
from collections import deque
import matplotlib.gridspec as gridspec

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Grid parameters (match C++ defines)
IMAX = 100
JMAX = 100
DS = 0.05  # meters per cell

# Parameters from semantic_safety.launch.py
DH0_OBSTACLE = 0.5
DH0_HUMAN = 1.5
SOCIAL_TANGENT_BIAS = 1.5
SOCIAL_TANGENT_LAYERS = 4

# Visualization options
ARROW_VIS_SCALE = 0.3       # Base arrow length scale factor
ARROW_DH0_SCALE = True      # If True, scale arrow length by dh0 (human arrows 3x longer)
                            # If False, all arrows same length (just show direction)
COLOR_ARROWS_BY_DEPTH = True


def create_combined_occupancy(obstacle_center, obstacle_radius, human_center, human_radius):
    """
    Create occupancy grid with both obstacle and human.
    Returns occupancy grid and masks for each obstacle type.
    """
    occ = np.ones((IMAX, JMAX), dtype=np.float32)
    is_obstacle = np.zeros((IMAX, JMAX), dtype=bool)
    is_human = np.zeros((IMAX, JMAX), dtype=bool)
    
    for i in range(IMAX):
        for j in range(JMAX):
            # Check obstacle
            dist_obs = np.sqrt((i - obstacle_center[0])**2 + (j - obstacle_center[1])**2)
            if dist_obs < obstacle_radius:
                occ[i, j] = -1.0
                is_obstacle[i, j] = True
            
            # Check human
            dist_human = np.sqrt((i - human_center[0])**2 + (j - human_center[1])**2)
            if dist_human < human_radius:
                occ[i, j] = -1.0
                is_human[i, j] = True
    
    return occ, is_obstacle, is_human


def find_boundary(occ):
    """Find boundary cells (free cells adjacent to occupied cells)."""
    bound = occ.copy()
    
    # Set grid borders as boundary
    bound[0, :] = 0.0
    bound[IMAX-1, :] = 0.0
    bound[:, 0] = 0.0
    bound[:, JMAX-1] = 0.0
    
    # Find internal boundaries
    b0 = bound.copy()
    for i in range(1, IMAX-1):
        for j in range(1, JMAX-1):
            if b0[i, j] == 1.0:
                neighbors = [
                    b0[i+1, j], b0[i-1, j], b0[i, j+1], b0[i, j-1],
                    b0[i+1, j+1], b0[i-1, j+1], b0[i+1, j-1], b0[i-1, j-1]
                ]
                if any(n == -1.0 for n in neighbors):
                    bound[i, j] = 0.0
    
    return bound


def compute_boundary_gradients_combined(bound, is_obstacle_mask, is_human_mask):
    """
    Compute boundary gradients with different dh0 for obstacle vs human boundaries.
    """
    guidance_x = np.zeros((IMAX, JMAX), dtype=np.float32)
    guidance_y = np.zeros((IMAX, JMAX), dtype=np.float32)
    obstacle_boundary_cells = []
    human_boundary_cells = []
    
    DH0_BORDER = 1.0
    
    # Set border gradients
    for i in range(IMAX):
        for j in range(JMAX):
            if i == 0:
                guidance_x[i, j] = DH0_BORDER
            if i == IMAX - 1:
                guidance_x[i, j] = -DH0_BORDER
            if j == 0:
                guidance_y[i, j] = DH0_BORDER
            if j == JMAX - 1:
                guidance_y[i, j] = -DH0_BORDER
    
    # Compute gradients at internal boundary cells
    for i in range(1, IMAX-1):
        for j in range(1, JMAX-1):
            if bound[i, j] == 0.0:
                gx = 0.0
                gy = 0.0
                for p in range(-1, 2):
                    for q in range(-1, 2):
                        if q > 0:
                            gx += bound[i+q, j+p]
                            gy += bound[i+p, j+q]
                        elif q < 0:
                            gx -= bound[i+q, j+p]
                            gy -= bound[i+p, j+q]
                guidance_x[i, j] = gx
                guidance_y[i, j] = gy
    
    # Normalize and scale based on obstacle type
    for i in range(1, IMAX-1):
        for j in range(1, JMAX-1):
            if bound[i, j] == 0.0:
                V = np.sqrt(guidance_x[i, j]**2 + guidance_y[i, j]**2)
                if V != 0.0:
                    guidance_x[i, j] /= V
                    guidance_y[i, j] /= V
                
                # Determine which obstacle this boundary belongs to
                # Check 8-connected neighbors for obstacle type
                is_human_boundary = False
                is_obstacle_boundary = False
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < IMAX and 0 <= nj < JMAX:
                            if is_human_mask[ni, nj]:
                                is_human_boundary = True
                            if is_obstacle_mask[ni, nj]:
                                is_obstacle_boundary = True
                
                # Use appropriate dh0
                if is_human_boundary:
                    local_dh0 = DH0_HUMAN
                    human_boundary_cells.append((i, j))
                elif is_obstacle_boundary:
                    local_dh0 = DH0_OBSTACLE
                    obstacle_boundary_cells.append((i, j))
                else:
                    local_dh0 = DH0_OBSTACLE  # default for border
                
                guidance_x[i, j] *= local_dh0
                guidance_y[i, j] *= local_dh0
    
    return guidance_x, guidance_y, obstacle_boundary_cells, human_boundary_cells


def red_black_sor_solve(grid, force, bound, w_SOR, max_epochs=100, iters_per_epoch=20, tol=1e-4):
    """Red-Black SOR solver."""
    h = grid.copy().astype(np.float64)
    f = force.astype(np.float64)
    
    rows, cols = np.indices((IMAX, JMAX))
    red_mask = ((rows % 2) == (cols % 2))
    black_mask = ((rows % 2) != (cols % 2))
    update_mask = (bound != 0.0)
    
    red_update = red_mask & update_mask
    black_update = black_mask & update_mask
    
    for epoch in range(max_epochs):
        for it in range(iters_per_epoch - 1):
            neighbors = (
                np.roll(h, -1, axis=0) +
                np.roll(h, +1, axis=0) +
                np.roll(h, -1, axis=1) +
                np.roll(h, +1, axis=1)
            )
            dg = 0.25 * (neighbors - f) - h
            h = np.where(red_update, h + w_SOR * dg, h)
            
            neighbors = (
                np.roll(h, -1, axis=0) +
                np.roll(h, +1, axis=0) +
                np.roll(h, -1, axis=1) +
                np.roll(h, +1, axis=1)
            )
            dg = 0.25 * (neighbors - f) - h
            h = np.where(black_update, h + w_SOR * dg, h)
        
        neighbors = (
            np.roll(h, -1, axis=0) +
            np.roll(h, +1, axis=0) +
            np.roll(h, -1, axis=1) +
            np.roll(h, +1, axis=1)
        )
        dg = 0.25 * (neighbors - f) - h
        h = np.where(update_mask, h + dg, h)
        
        rss = np.sqrt(np.sum(dg[update_mask]**2)) * DS
        if rss < tol:
            break
    
    return h.astype(np.float32)


def solve_laplace(guidance_x, guidance_y, bound, max_epochs=100, tol=1e-4):
    """Solve Laplace equation for guidance field smoothing."""
    N = IMAX // 5
    w_SOR = 2.0 / (1.0 + np.sin(np.pi / (N + 1)))
    f0 = np.zeros((IMAX, JMAX), dtype=np.float32)
    
    gx = red_black_sor_solve(guidance_x, f0, bound, w_SOR, max_epochs=max_epochs, tol=tol)
    gy = red_black_sor_solve(guidance_y, f0, bound, w_SOR, max_epochs=max_epochs, tol=tol)
    
    return gx, gy


def apply_social_tangent_post_solve(guidance_x, guidance_y, bound, 
                                     human_boundary_cells, 
                                     social_tangent_layers=SOCIAL_TANGENT_LAYERS,
                                     social_tangent_bias=SOCIAL_TANGENT_BIAS):
    """Apply tangential bias after Laplace solve for human obstacles."""
    gx = guidance_x.copy()
    gy = guidance_y.copy()
    
    if len(human_boundary_cells) == 0 or social_tangent_layers <= 0:
        return gx, gy, np.zeros((IMAX, JMAX), dtype=np.int8)
    
    dist_from_human = np.full((IMAX, JMAX), np.inf)
    queue = deque()
    
    for (i, j) in human_boundary_cells:
        dist_from_human[i, j] = 0
        queue.append((i, j))
    
    while queue:
        ci, cj = queue.popleft()
        cur_dist = dist_from_human[ci, cj]
        if cur_dist >= social_tangent_layers:
            continue
        
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = ci + di, cj + dj
            if 1 <= ni < IMAX-1 and 1 <= nj < JMAX-1:
                if dist_from_human[ni, nj] == np.inf:
                    dist_from_human[ni, nj] = cur_dist + 1
                    queue.append((ni, nj))
    
    tangent_layer_display = np.zeros((IMAX, JMAX), dtype=np.int8)
    
    for i in range(1, IMAX-1):
        for j in range(1, JMAX-1):
            dist = dist_from_human[i, j]
            if 1 <= dist <= social_tangent_layers and bound[i, j] > 0:
                gx_val = gx[i, j]
                gy_val = gy[i, j]
                mag = np.sqrt(gx_val**2 + gy_val**2)
                if mag < 0.01:
                    continue
                
                gx_norm = gx_val / mag
                gy_norm = gy_val / mag
                
                sign = -1.0
                decay = 1.0 - (dist - 1) / social_tangent_layers
                
                biased_gx = gx_norm + sign * social_tangent_bias * decay * (-gy_norm)
                biased_gy = gy_norm + sign * social_tangent_bias * decay * gx_norm
                
                V2 = np.sqrt(biased_gx**2 + biased_gy**2)
                if V2 > 0:
                    gx[i, j] = biased_gx / V2 * mag
                    gy[i, j] = biased_gy / V2 * mag
                
                tangent_layer_display[i, j] = 1
    
    return gx, gy, tangent_layer_display


def compute_forcing_function(guidance_x, guidance_y, bound):
    """Compute forcing function from divergence of guidance field."""
    max_div = 10.0
    
    dgx_di = (np.roll(guidance_x, -1, axis=0) - np.roll(guidance_x, 1, axis=0)) / (2.0 * DS)
    dgy_dj = (np.roll(guidance_y, -1, axis=1) - np.roll(guidance_y, 1, axis=1)) / (2.0 * DS)
    div = dgx_di + dgy_dj
    
    force = np.zeros((IMAX, JMAX), dtype=np.float32)
    force = np.where(bound > 0, div, force)
    force = np.where(bound < 0, np.clip(div, 0.0, max_div), force)
    
    return force * DS * DS


def solve_poisson(h_init, force, bound, max_epochs=100, tol=1e-4):
    """Solve Poisson equation for safety function."""
    N = IMAX // 5
    w_SOR = 2.0 / (1.0 + np.sin(np.pi / (N + 1)))
    
    h = red_black_sor_solve(h_init, force, bound, w_SOR, max_epochs=max_epochs, tol=tol)
    
    return h


def simulate_robot_path(start, goal, h, guidance_x, guidance_y, bound, 
                        is_obstacle, is_human, dt=0.01, max_steps=2000,
                        goal_threshold=0.1):
    """
    Simulate robot path using safety filter.
    
    The safety filter modifies velocity to ensure h stays positive:
    v = v_nom + lambda * dh/dx
    where lambda ensures dh/dt + gamma * h >= 0
    
    Args:
        start: (x, y) start position in meters
        goal: (x, y) goal position in meters
        h: safety function grid
        guidance_x, guidance_y: guidance field (smoothed gradient of h)
        bound: boundary grid
        is_obstacle, is_human: obstacle masks
        dt: simulation timestep
        max_steps: maximum simulation steps
        goal_threshold: distance to goal to stop
    
    Returns:
        path: list of (x, y) positions
    """
    path = [start]
    x, y = start
    
    # Control parameters (from C++ safety_filter)
    gamma = 1.0  # wn in C++
    v_max = 0.5  # max velocity
    sigma_sontag = 1.0
    
    for step in range(max_steps):
        # Check if reached goal
        dist_to_goal = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
        if dist_to_goal < goal_threshold:
            break
        
        # Convert position to grid indices
        j = x / DS  # x -> j (column)
        i = y / DS  # y -> i (row)
        
        # Check bounds
        if i < 1 or i >= IMAX - 1 or j < 1 or j >= JMAX - 1:
            break
        
        # Bilinear interpolation for h value
        i0, j0 = int(i), int(j)
        i1, j1 = min(i0 + 1, IMAX - 1), min(j0 + 1, JMAX - 1)
        fi, fj = i - i0, j - j0
        
        h_val = (1 - fi) * (1 - fj) * h[i0, j0] + \
                (1 - fi) * fj * h[i0, j1] + \
                fi * (1 - fj) * h[i1, j0] + \
                fi * fj * h[i1, j1]
        
        # Get guidance field (this is the smoothed gradient v)
        # guidance_x is i-direction (y), guidance_y is j-direction (x)
        vx = (1 - fi) * (1 - fj) * guidance_y[i0, j0] + \
             (1 - fi) * fj * guidance_y[i0, j1] + \
             fi * (1 - fj) * guidance_y[i1, j0] + \
             fi * fj * guidance_y[i1, j1]
        vy = (1 - fi) * (1 - fj) * guidance_x[i0, j0] + \
             (1 - fi) * fj * guidance_x[i0, j1] + \
             fi * (1 - fj) * guidance_x[i1, j0] + \
             fi * fj * guidance_x[i1, j1]
        
        # Nominal velocity: towards goal
        dx = goal[0] - x
        dy = goal[1] - y
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 0.01:
            vd_x = v_max * dx / dist
            vd_y = v_max * dy / dist
        else:
            vd_x, vd_y = 0, 0
        
        # Safety filter (simplified from C++ safety_filter)
        # dh/dx for control uses guidance field
        dhdx = vx
        dhdy = vy
        
        # Activating function: a = v · k_nom + gamma * h
        a = dhdx * vd_x + dhdy * vd_y + gamma * h_val
        
        # b = ||dh||^2 / P
        Pu = 2.0
        b = (dhdx**2 + dhdy**2) / Pu
        
        # Sontag formula for lambda
        if b > 1e-4:
            lambda_sf = 0.5 * (-a + np.sqrt(a**2 + sigma_sontag * b**2)) / b
        else:
            lambda_sf = 0.0
        
        # Apply safety filter
        v_x = vd_x + lambda_sf * dhdx / Pu
        v_y = vd_y + lambda_sf * dhdy / Pu
        
        # Saturate velocity
        v_mag = np.sqrt(v_x**2 + v_y**2)
        if v_mag > v_max:
            v_x = v_max * v_x / v_mag
            v_y = v_max * v_y / v_mag
        
        # Update position
        x += v_x * dt
        y += v_y * dt
        path.append((x, y))
    
    return path


def plot_combined_scene():
    """
    Create a 2-row plot with obstacle on left, human on right in the SAME field:
    - Row 0: 2D Guidance Field + Safety Function + Robot Path
    - Row 1: 3D Poisson Safety Function Surface
    """
    fig = plt.figure(figsize=(10, 14))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.8], hspace=0.25)
    
    # Place human on left, obstacle on right with 1.2m separation
    # Center (2.5, 2.5). Gap 1.2m => +/- 0.6m
    # Human x = 1.9, Obstacle x = 3.1
    human_center = (IMAX // 2, int(1.9 / DS))         # Left side (y=2.5, x=1.9)
    obstacle_center = (IMAX // 2, int(3.1 / DS))      # Right side (y=2.5, x=3.1)
    radius = 8  # cells (same for both)
    
    # Create combined occupancy
    occ, is_obstacle, is_human = create_combined_occupancy(
        obstacle_center, radius, human_center, radius
    )
    
    # Find boundaries
    bound = find_boundary(occ)
    
    # Compute boundary gradients with different dh0
    guidance_x, guidance_y, obs_boundary, human_boundary = compute_boundary_gradients_combined(
        bound, is_obstacle, is_human
    )
    
    # Store PRE-LAPLACE boundary gradients for layer 0 visualization (true normals)
    boundary_gx = guidance_x.copy()
    boundary_gy = guidance_y.copy()
    
    # Solve Laplace for smooth guidance
    guidance_x, guidance_y = solve_laplace(guidance_x, guidance_y, bound)
    
    # Apply social tangent only around human
    guidance_x, guidance_y, tangent_display = apply_social_tangent_post_solve(
        guidance_x, guidance_y, bound, human_boundary,
        SOCIAL_TANGENT_LAYERS, SOCIAL_TANGENT_BIAS
    )
    
    # Compute forcing and solve Poisson
    force = compute_forcing_function(guidance_x, guidance_y, bound)
    h_init = np.zeros((IMAX, JMAX), dtype=np.float32)
    for i in range(IMAX):
        for j in range(JMAX):
            if bound[i, j] <= 0:
                h_init[i, j] = 0.0
    h = solve_poisson(h_init, force, bound)
    
    # Colormap
    # Colormap (Viridis samples)
    viridis = plt.get_cmap('viridis')
    colors = viridis(np.linspace(0, 1, 256))
    cmap = LinearSegmentedColormap.from_list('safety', colors)
    
    # Meshgrid for 3D
    Y_grid, X_grid = np.mgrid[0:IMAX, 0:JMAX]
    X_grid = X_grid * DS
    Y_grid = Y_grid * DS
    
    # --- ROW 0: 2D PLOT ---
    ax_2d = fig.add_subplot(gs[0])
    extent = [0, JMAX * DS, 0, IMAX * DS]
    im = ax_2d.imshow(h, origin='lower', cmap=cmap, vmin=0, vmax=1.2, extent=extent)
    
    # Plot obstacle (gray)
    circle_obs = Circle((obstacle_center[1] * DS, obstacle_center[0] * DS), radius * DS,
                         color='#888888', alpha=0.9, zorder=10)
    ax_2d.add_patch(circle_obs)
    ax_2d.text(obstacle_center[1] * DS, obstacle_center[0] * DS, 'O',
               fontsize=16, ha='center', va='center', color='white',
               fontweight='bold', zorder=11)
    
    # Plot human (red)
    circle_human = Circle((human_center[1] * DS, human_center[0] * DS), radius * DS,
                           color='red', alpha=0.9, zorder=10)
    ax_2d.add_patch(circle_human)
    ax_2d.text(human_center[1] * DS, human_center[0] * DS, 'H',
               fontsize=16, ha='center', va='center', color='white',
               fontweight='bold', zorder=11)
    
    # Arrow colors - same unified palette as plot_guidance_field_comparison.py
    layer_colors = ['white', '#FFD700', '#FFA500', '#FFCC80', '#FFE0B2', 'cyan']
    
    # --- QUIVER ON REGULAR GRID (Actual Guidance Field) ---
    # Sample the computed guidance field on a regular grid
    skip = 5  # Sample every 5th cell for cleaner look
    
    # Create meshgrid for arrow positions (in meters)
    arrow_i = np.arange(skip, IMAX - skip, skip)
    arrow_j = np.arange(skip, JMAX - skip, skip)
    J_grid, I_grid = np.meshgrid(arrow_j, arrow_i)
    
    # Sample guidance field at grid points
    GX = guidance_x[arrow_i][:, arrow_j]  # i-direction (y in plot)
    GY = guidance_y[arrow_i][:, arrow_j]  # j-direction (x in plot)
    
    # Mask out arrows inside obstacles and near human boundary (layers 0-1)
    layer1_radius = radius + 6  # Mask layer 0 and 1 region
    mask = np.ones_like(GX, dtype=bool)
    for ii, i_idx in enumerate(arrow_i):
        for jj, j_idx in enumerate(arrow_j):
            if bound[i_idx, j_idx] <= 0:
                mask[ii, jj] = False
            # Mask near human for separate layer visualization
            dist_to_human = np.sqrt((i_idx - human_center[0])**2 + (j_idx - human_center[1])**2)
            if dist_to_human <= layer1_radius:
                mask[ii, jj] = False
    
    # Keep actual magnitude for arrow length
    magnitude = np.sqrt(GX**2 + GY**2)
    magnitude[magnitude < 1e-6] = 1e-6  # Avoid division by zero
    
    # Apply mask (set to NaN inside obstacles)
    GX_plot = np.where(mask, GX, np.nan)
    GY_plot = np.where(mask, GY, np.nan)
    
    # Arrow scale factor
    arrow_scale = ARROW_VIS_SCALE * 2.5
    
    # Plot quiver with magnitude-based coloring
    quiv = ax_2d.quiver(
        J_grid * DS, I_grid * DS,  # x, y positions
        GY_plot * arrow_scale, GX_plot * arrow_scale,  # u, v components
        magnitude,  # color by magnitude
        cmap='YlOrRd',
        alpha=0.9,
        scale=15,
        width=0.005,
        headwidth=4,
        headlength=5,
        zorder=5
    )
    
    # --- LAYER 0: Pure Normal (boundary layer, no tangent bias) ---
    # Use circular sampling for consistent coverage around human
    n_arrows_layer0 = 14
    layer0_radius = radius + 2  # Just outside boundary
    angles = np.linspace(0, 2 * np.pi, n_arrows_layer0, endpoint=False)
    
    for angle in angles:
        i = int(human_center[0] + layer0_radius * np.cos(angle))
        j = int(human_center[1] + layer0_radius * np.sin(angle))
        
        if not (1 <= i < IMAX - 1 and 1 <= j < JMAX - 1):
            continue
        if bound[i, j] <= 0:
            continue
        
        # Pure radial normal (NO tangent bias)
        di = i - human_center[0]
        dj = j - human_center[1]
        r = np.sqrt(di**2 + dj**2)
        if r < 0.01:
            continue
        nx = di / r
        ny = dj / r
        
        arrow_len = DH0_HUMAN * arrow_scale * 0.4
        ax_2d.quiver(j * DS, i * DS, ny * arrow_len, nx * arrow_len,
                     color='#FD8D3C', alpha=0.95, scale=15, width=0.005,
                     zorder=10, headwidth=4, headlength=5)
    
    # --- LAYER 1+: With Tangent Bias ---
    # Second ring further out showing the tangent bias
    n_arrows_layer1 = 18
    layer1_radius = radius + 5  # Layer 1-2 region
    angles = np.linspace(0, 2 * np.pi, n_arrows_layer1, endpoint=False) + np.pi / 4  # CCW offset by 1/8
    
    for angle in angles:
        i = int(human_center[0] + layer1_radius * np.cos(angle))
        j = int(human_center[1] + layer1_radius * np.sin(angle))
        
        if not (1 <= i < IMAX - 1 and 1 <= j < JMAX - 1):
            continue
        if bound[i, j] <= 0:
            continue
        
        # Radial direction
        di = i - human_center[0]
        dj = j - human_center[1]
        r = np.sqrt(di**2 + dj**2)
        if r < 0.01:
            continue
        nx = di / r
        ny = dj / r
        
        # Apply tangent bias (CCW)
        tx = -ny
        ty = nx
        layer_idx = 1
        decay = 1.0 - layer_idx / SOCIAL_TANGENT_LAYERS
        sign = -1.0
        biased_x = nx + sign * SOCIAL_TANGENT_BIAS * decay * tx
        biased_y = ny + sign * SOCIAL_TANGENT_BIAS * decay * ty
        bm = np.sqrt(biased_x**2 + biased_y**2)
        gx_val = biased_x / bm
        gy_val = biased_y / bm
        
        arrow_len = DH0_HUMAN * arrow_scale * 0.35
        ax_2d.quiver(j * DS, i * DS, gy_val * arrow_len, gx_val * arrow_len,
                     color='#FECC5C', alpha=0.95, scale=15, width=0.005,
                     zorder=9, headwidth=4, headlength=5)
    
    # --- PATH FLOW FIELD VISUALIZATION ---
    # --- PATH FLOW FIELD VISUALIZATION ---
    # Plot paths every 0.5m along X axis from y=0.5
    start_y = 0.5
    goal_y = 4.5
    # From 0.5 to 4.5 (inclusive)
    x_steps = np.arange(0.5, 5.0, 0.5)
    # Shift the third point (index 2) by 0.1
    if len(x_steps) > 2:
        x_steps[2] += 0.1
        x_steps[0] += 0.06
        x_steps[1] += 0.06
        x_steps[3] += 0.06
    print(x_steps)
    
    # Use a color cycle or map for distinct paths
    path_colors = plt.cm.jet(np.linspace(0, 1, len(x_steps)))
    
    print(f"Simulating {len(x_steps)} paths starting at y={start_y}...")
    
    for idx, start_x in enumerate(x_steps):
        start_pos = (start_x, start_y)
        goal_pos = (start_x, goal_y)
        
        # Simulate path
        path = simulate_robot_path(start_pos, goal_pos, h, guidance_x, guidance_y, 
                                   bound, is_obstacle, is_human)
        
        if len(path) > 1:
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            
            # Plot path
            label = f'Start x={start_x:.1f}' if idx % 2 == 0 else None
            ax_2d.plot(px, py, '-', color=path_colors[idx], linewidth=2, zorder=15, 
                       alpha=0.9, label=label)
            
            # Mark start
            ax_2d.plot(start_pos[0], start_pos[1], 'o', color=path_colors[idx], 
                       markersize=8, zorder=16, markeredgecolor='white', markeredgewidth=1)
            # Mark goal
            ax_2d.plot(goal_pos[0], goal_pos[1], '*', color=path_colors[idx], 
                       markersize=12, zorder=16, markeredgecolor='white', markeredgewidth=1)

    print("Path simulation complete.")

    ax_2d.set_xlim(0, JMAX * DS)
    ax_2d.set_ylim(0, IMAX * DS)
    ax_2d.set_xlabel('X (m)', fontsize=12)
    ax_2d.set_ylabel('Y (m)', fontsize=12)
    ax_2d.set_title(f'2D Guidance Field + Safety Function\n(Obstacle: dh0={DH0_OBSTACLE}, Human: dh0={DH0_HUMAN})', 
                    fontsize=13, fontweight='bold')
    ax_2d.set_aspect('equal')
    
    # Add legend - outside the plot
    legend_text = (f"O = Obstacle (gray, dh0={DH0_OBSTACLE})\n"
                   f"H = Human (red, dh0={DH0_HUMAN})\n"
                   f"Social tangent: layers={SOCIAL_TANGENT_LAYERS}, bias={SOCIAL_TANGENT_BIAS}")
    fig.text(0.02, 0.95, legend_text, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    # Move path legend to bottom of plot
    ax_2d.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), fontsize=8, ncol=3)
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax_2d, fraction=0.046, pad=0.04)
    cbar.set_label('h(x) [m]', fontsize=11)
    
    # --- ROW 1: 3D SURFACE ---
    ax_3d = fig.add_subplot(gs[1], projection='3d')
    ax_3d.plot_surface(X_grid, Y_grid, h, cmap=cmap, linewidth=0, antialiased=False, vmin=0, vmax=1.2)
    ax_3d.set_title('3D Poisson Safety Function (Combined Scene)', fontsize=12, fontweight='bold')
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('h(x)')
    ax_3d.set_zlim(0, 1.2)
    ax_3d.view_init(elev=35, azim=-135)
    
    plt.suptitle('Obstacle + Human in Same Poisson Field', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    output_path = os.path.join(SCRIPT_DIR, 'obstacle_human_combined.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    output_path_pdf = os.path.join(SCRIPT_DIR, 'obstacle_human_combined.pdf')
    plt.savefig(output_path_pdf, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    print("Generating combined Obstacle + Human scene...")
    print(f"Grid: {IMAX}x{JMAX}, Cell size: {DS}m")
    print(f"dh0_obstacle: {DH0_OBSTACLE}, dh0_human: {DH0_HUMAN}")
    print(f"Social tangent: layers={SOCIAL_TANGENT_LAYERS}, bias={SOCIAL_TANGENT_BIAS}")
    print()
    
    plot_combined_scene()

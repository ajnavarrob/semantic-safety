#!/usr/bin/env python3
"""
Visualization: Guidance Field and Poisson Safety Function Comparison

Demonstrates the difference between:
- Left: Normal circular obstacle (no inflation, normal gradients everywhere)
- Right: Human obstacle (no inflation, boundary layer normal, 
         social tangential bias from layer 1 onwards)

Follows the exact approach from semantics_poisson.cpp:
1. No inflation of obstacle boundaries
2. Boundary layer (layer 0) vectors are ALWAYS normal to preserve CBF guarantees
3. Social tangential bias is applied only from layer 1 onwards via BFS
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless rendering

# Set global font to Times New Roman
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman'] + matplotlib.rcParams['font.serif']

import numpy as np
import matplotlib.pyplot as plt
import os

# Output directory (same as script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import distance_transform_edt, binary_dilation, label
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import spsolve
from collections import deque
import matplotlib.gridspec as gridspec

# Match C++ defines from poisson.h
IMAX = 100  # Grid rows (i direction = y)
JMAX = 100  # Grid cols (j direction = x)
DS = 0.05   # meters per cell

# Parameters from semantic_safety.launch.py (ROS launch file defaults)
DH0_OBSTACLE = 0.5  # Gradient magnitude at obstacle boundaries
DH0_HUMAN = 1.5     # Gradient magnitude at human boundaries (larger = more repulsive)
SOCIAL_TANGENT_BIAS = 1.5  # tangential bias strength (beta) - 1.5 gives strong leftward curve
SOCIAL_TANGENT_LAYERS = 4  # number of layers outward from boundary with tangent bias
ARROW_VIS_SCALE = 0.3      # Scale arrow lengths for clearer visualization
COLOR_ARROWS_BY_DEPTH = True # Color arrows by distance (Cyan/Gold at boundary -> White away)

def create_circular_obstacle(center_ij, radius_cells):
    """
    Create occupancy grid with a circular obstacle.
    
    Args:
        center_ij: (i, j) center in grid coordinates
        radius_cells: radius in cells
    
    Returns:
        occ: IMAX x JMAX grid where -1=occupied, 1=free
    """
    occ = np.ones((IMAX, JMAX), dtype=np.float32)
    for i in range(IMAX):
        for j in range(JMAX):
            dist = np.sqrt((i - center_ij[0])**2 + (j - center_ij[1])**2)
            if dist < radius_cells:
                occ[i, j] = -1.0
    return occ


def find_boundary(occ):
    """
    Find boundary cells (free cells adjacent to occupied cells).
    Matches semantics_poisson.cpp find_boundary logic.
    
    Returns:
        bound: -1=occupied, 0=boundary, 1=free space
    """
    bound = occ.copy()
    
    # Set grid borders as boundary
    bound[0, :] = 0.0
    bound[IMAX-1, :] = 0.0
    bound[:, 0] = 0.0
    bound[:, JMAX-1] = 0.0
    
    # Find internal boundaries (free cells adjacent to occupied cells)
    b0 = bound.copy()
    for i in range(1, IMAX-1):
        for j in range(1, JMAX-1):
            if b0[i, j] == 1.0:  # free cell
                # Check 8-connected neighbors for occupied
                neighbors = [
                    b0[i+1, j], b0[i-1, j], b0[i, j+1], b0[i, j-1],
                    b0[i+1, j+1], b0[i-1, j+1], b0[i+1, j-1], b0[i-1, j-1]
                ]
                if any(n == -1.0 for n in neighbors):
                    bound[i, j] = 0.0  # Mark as boundary
    
    return bound


def compute_boundary_gradients(bound, is_human=False, dh0=None):
    """
    Compute boundary gradients (guidance field at boundaries).
    Matches semantics_poisson.cpp compute_boundary_gradients logic EXACTLY.
    
    Args:
        bound: boundary array from find_boundary (-1=occupied, 0=boundary, 1=free)
        is_human: if True, use DH0_HUMAN for internal boundary magnitude
        dh0: override gradient magnitude (if None, use DH0_OBSTACLE or DH0_HUMAN)
    
    Returns:
        guidance_x, guidance_y: gradient components at each cell
        boundary_cells: list of (i, j) internal boundary cell coordinates
    """
    guidance_x = np.zeros((IMAX, JMAX), dtype=np.float32)
    guidance_y = np.zeros((IMAX, JMAX), dtype=np.float32)
    boundary_cells = []
    
    # Determine the dh0 for internal obstacle boundaries
    if dh0 is None:
        local_dh0 = DH0_HUMAN if is_human else DH0_OBSTACLE
    else:
        local_dh0 = dh0
    
    # dh0 = 1.0 is used for the OUTER BORDER gradients (matching C++ class constant)
    DH0_BORDER = 1.0
    
    # Set BORDER gradients (outer edge of grid) - uses dh0=1.0 constant
    # These point INWARD into the grid
    for i in range(IMAX):
        for j in range(JMAX):
            if i == 0:
                guidance_x[i, j] = DH0_BORDER  # Pointing in +i direction (into grid)
            if i == IMAX - 1:
                guidance_x[i, j] = -DH0_BORDER  # Pointing in -i direction (into grid)
            if j == 0:
                guidance_y[i, j] = DH0_BORDER  # Pointing in +j direction (into grid)
            if j == JMAX - 1:
                guidance_y[i, j] = -DH0_BORDER  # Pointing in -j direction (into grid)
    
    # Compute gradients at INTERNAL boundary cells (bound == 0)
    # This uses the 3x3 kernel exactly as in semantics_poisson.cpp
    # The kernel computes gradient pointing AWAY from occupied cells (into free space)
    for i in range(1, IMAX-1):
        for j in range(1, JMAX-1):
            if bound[i, j] == 0.0:  # boundary cell
                gx = 0.0
                gy = 0.0
                # Exact C++ loop structure:
                # for(int p = -1; p <= 1; p++){
                #     for(int q = -1; q <= 1; q++){
                #         if(q > 0){
                #             guidance_x[i*JMAX+j] += bound[(i+q)*JMAX+(j+p)];
                #             guidance_y[i*JMAX+j] += bound[(i+p)*JMAX+(j+q)];
                #         }
                #         else if (q < 0){
                #             guidance_x[i*JMAX+j] -= bound[(i+q)*JMAX+(j+p)];
                #             guidance_y[i*JMAX+j] -= bound[(i+p)*JMAX+(j+q)];
                #         }
                #     }
                # }
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
    
    # Normalize INTERNAL boundary gradients and scale by local_dh0
    # (Outer border gradients are already set and don't need this)
    for i in range(IMAX):
        for j in range(JMAX):
            if bound[i, j] == 0.0:  # boundary cell
                V = np.sqrt(guidance_x[i, j]**2 + guidance_y[i, j]**2)
                if V != 0.0:
                    guidance_x[i, j] /= V
                    guidance_y[i, j] /= V
                # Scale by local_dh0 (DH0_HUMAN or DH0_OBSTACLE)
                guidance_x[i, j] *= local_dh0
                guidance_y[i, j] *= local_dh0
                boundary_cells.append((i, j))
    
    return guidance_x, guidance_y, boundary_cells


def red_black_sor_solve(grid, force, bound, w_SOR, max_epochs=100, iters_per_epoch=20, tol=1e-4):
    """
    Red-Black SOR solver matching kernel.cu exactly.
    
    Key points from CUDA implementation:
    - boundary[i] == 0.0f means SKIP (Dirichlet BC - keep fixed value)
    - boundary[i] != 0.0f means UPDATE (both free space and occupied)
    - Red cells: (row % 2) == (col % 2)
    - Black cells: (row % 2) != (col % 2)
    - Update formula: grid[i] += w_SOR * (0.25 * (neighbors - force) - grid[i])
    """
    h = grid.copy().astype(np.float64)  # Use float64 for numerical stability
    f = force.astype(np.float64)
    
    # Create red and black masks
    rows, cols = np.indices((IMAX, JMAX))
    red_mask = ((rows % 2) == (cols % 2))
    black_mask = ((rows % 2) != (cols % 2))
    
    # Mask for cells to update: bound != 0 (NOT the boundary Dirichlet cells)
    update_mask = (bound != 0.0)
    
    red_update = red_mask & update_mask
    black_update = black_mask & update_mask
    
    for epoch in range(max_epochs):
        for it in range(iters_per_epoch - 1):
            # Update red cells
            neighbors = (
                np.roll(h, -1, axis=0) +  # i+1
                np.roll(h, +1, axis=0) +  # i-1
                np.roll(h, -1, axis=1) +  # j+1
                np.roll(h, +1, axis=1)    # j-1
            )
            dg = 0.25 * (neighbors - f) - h
            h = np.where(red_update, h + w_SOR * dg, h)
            
            # Update black cells
            neighbors = (
                np.roll(h, -1, axis=0) +
                np.roll(h, +1, axis=0) +
                np.roll(h, -1, axis=1) +
                np.roll(h, +1, axis=1)
            )
            dg = 0.25 * (neighbors - f) - h
            h = np.where(black_update, h + w_SOR * dg, h)
        
        # Compute residual (one more Gauss-Seidel iteration)
        neighbors = (
            np.roll(h, -1, axis=0) +
            np.roll(h, +1, axis=0) +
            np.roll(h, -1, axis=1) +
            np.roll(h, +1, axis=1)
        )
        dg = 0.25 * (neighbors - f) - h
        h = np.where(update_mask, h + dg, h)
        
        # Compute RSS (only on updated cells)
        rss = np.sqrt(np.sum(dg[update_mask]**2)) * DS
        if rss < tol:
            break
    
    return h.astype(np.float32)


def solve_laplace(guidance_x, guidance_y, bound, max_epochs=100, tol=1e-4):
    """
    Solve Laplace equation nabla^2 g = 0 for guidance field smoothing.
    Uses the same poissonSolve from kernel.cu with f=0.
    
    Boundary cells (bound==0) hold fixed Dirichlet BC values (the gradient values we computed).
    """
    # SOR parameter (matches C++ code)
    N = IMAX // 5
    w_SOR = 2.0 / (1.0 + np.sin(np.pi / (N + 1)))
    
    # f0 = zeros (Laplace equation has no forcing term)
    f0 = np.zeros((IMAX, JMAX), dtype=np.float32)
    
    gx = red_black_sor_solve(guidance_x, f0, bound, w_SOR, max_epochs=max_epochs, tol=tol)
    gy = red_black_sor_solve(guidance_y, f0, bound, w_SOR, max_epochs=max_epochs, tol=tol)
    
    return gx, gy


def apply_social_tangent_post_solve(guidance_x, guidance_y, bound, 
                                     human_boundary_cells, 
                                     social_tangent_layers=SOCIAL_TANGENT_LAYERS,
                                     social_tangent_bias=SOCIAL_TANGENT_BIAS):
    """
    Apply tangential bias AFTER Laplace solve, to free space cells near humans.
    Matches semantics_poisson.cpp post-solve tangent application.
    
    Key points (from C++ code):
    - Layer 0 (boundary cells where bound==0) KEEP normal gradients (CBF guarantee)
    - Layers 1 to N (free space within N cells of human boundary) get tangent bias
    - Bias decays with distance from boundary
    """
    gx = guidance_x.copy()
    gy = guidance_y.copy()
    
    if len(human_boundary_cells) == 0 or social_tangent_layers <= 0:
        return gx, gy, np.zeros((IMAX, JMAX), dtype=np.int8)
    
    # BFS to compute distance from human boundary cells
    dist_from_human = np.full((IMAX, JMAX), np.inf)
    queue = deque()
    
    # Seed with human boundary cells (layer 0)
    for (i, j) in human_boundary_cells:
        dist_from_human[i, j] = 0
        queue.append((i, j))
    
    # BFS
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
    
    # Apply tangent bias to FREE SPACE cells (bound > 0) at distance 1 to N
    # Skip boundary cells (bound == 0) to preserve CBF normal!
    tangent_layer_display = np.zeros((IMAX, JMAX), dtype=np.int8)
    
    for i in range(1, IMAX-1):
        for j in range(1, JMAX-1):
            dist = dist_from_human[i, j]
            # Only apply to FREE SPACE (bound > 0) within range
            # Note: dist >= 1 means NOT the boundary itself
            if 1 <= dist <= social_tangent_layers and bound[i, j] > 0:
                gx_val = gx[i, j]
                gy_val = gy[i, j]
                mag = np.sqrt(gx_val**2 + gy_val**2)
                if mag < 0.01:
                    continue
                
                # Normalize
                gx_norm = gx_val / mag
                gy_norm = gy_val / mag
                
                # CCW tangent: rotate 90° counterclockwise
                # (gx, gy) -> (-gy, gx)
                # sign = -1 for CCW (robot passes on human's right, curves left)
                sign = -1.0
                
                # Decay bias with distance: full at layer 1, fades to 0 at layer N
                decay = 1.0 - (dist - 1) / social_tangent_layers
                
                biased_gx = gx_norm + sign * social_tangent_bias * decay * (-gy_norm)
                biased_gy = gy_norm + sign * social_tangent_bias * decay * gx_norm
                
                # Normalize and restore magnitude
                V2 = np.sqrt(biased_gx**2 + biased_gy**2)
                if V2 > 0:
                    gx[i, j] = biased_gx / V2 * mag
                    gy[i, j] = biased_gy / V2 * mag
                
                tangent_layer_display[i, j] = 1
    
    return gx, gy, tangent_layer_display


def compute_forcing_function(guidance_x, guidance_y, bound):
    """
    Compute forcing function from divergence of guidance field.
    Matches semantics_poisson.cpp compute_optimal_forcing_function.
    Vectorized for speed.
    """
    max_div = 10.0
    
    # Compute divergence using vectorized central differences
    dgx_di = (np.roll(guidance_x, -1, axis=0) - np.roll(guidance_x, 1, axis=0)) / (2.0 * DS)
    dgy_dj = (np.roll(guidance_y, -1, axis=1) - np.roll(guidance_y, 1, axis=1)) / (2.0 * DS)
    div = dgx_di + dgy_dj
    
    # Apply conditions based on bound
    force = np.zeros((IMAX, JMAX), dtype=np.float32)
    force = np.where(bound > 0, div, force)  # free space
    force = np.where(bound < 0, np.clip(div, 0.0, max_div), force)  # occupied
    # boundary (bound == 0) stays 0
    
    return force * DS * DS


def solve_poisson(h_init, force, bound, max_epochs=100, tol=1e-4):
    """
    Solve Poisson equation nabla^2 h = f with Dirichlet BC h=fixed at boundaries (bound==0).
    Uses the same poissonSolve from kernel.cu.
    
    h_init should have h=0 at boundary cells (bound==0).
    """
    # SOR parameter (matches C++ code)
    N = IMAX // 5
    w_SOR = 2.0 / (1.0 + np.sin(np.pi / (N + 1)))
    
    h = red_black_sor_solve(h_init, force, bound, w_SOR, max_epochs=max_epochs, tol=tol)
    
    return h


def generate_guidance_and_safety(occ, is_human=False, apply_social_tangent=False):
    """
    Main pipeline: generate guidance field and Poisson safety function.
    
    Matches semantics_poisson.cpp solve_poisson_safety_function logic:
    1. find_boundary (NO inflation)
    2. compute_boundary_gradients (normal only at boundaries)
    3. solve Laplace for smooth guidance field
    4. (optionally) apply social tangent to layers 1+ post-solve
    5. compute forcing function
    6. solve Poisson for safety function h
    """
    # Step 1: Find boundaries
    bound = find_boundary(occ)
    
    # Step 2: Compute boundary gradients
    dh0 = DH0_HUMAN if is_human else DH0_OBSTACLE
    guidance_x, guidance_y, boundary_cells = compute_boundary_gradients(bound, is_human, dh0)
    
    # Step 3: Solve Laplace to smooth guidance in free space
    guidance_x, guidance_y = solve_laplace(guidance_x, guidance_y, bound)
    
    # Step 4: Apply social tangent bias (only for human, and only to layers 1+)
    tangent_layer_display = np.zeros((IMAX, JMAX), dtype=np.int8)
    if apply_social_tangent and is_human:
        guidance_x, guidance_y, tangent_layer_display = apply_social_tangent_post_solve(
            guidance_x, guidance_y, bound, boundary_cells,
            SOCIAL_TANGENT_LAYERS, SOCIAL_TANGENT_BIAS
        )
    
    # Step 5: Compute forcing function
    force = compute_forcing_function(guidance_x, guidance_y, bound)
    
    # Step 6: Solve Poisson for safety function
    h_init = np.zeros((IMAX, JMAX), dtype=np.float32)
    # Set initial values at boundaries (h=0 at obstacle surfaces)
    for i in range(IMAX):
        for j in range(JMAX):
            if bound[i, j] <= 0:
                h_init[i, j] = 0.0
    
    h = solve_poisson(h_init, force, bound)
    
    return guidance_x, guidance_y, h, bound, boundary_cells, tangent_layer_display


def plot_comparison():
    """
    Create a 4-row comparison:
    - Row 0: 2D Guidance Field + Safety Function (Image)
    - Row 1: 3D Poisson Safety Function (Surface only)
    - Row 2: Flat plane projection with arrows above surface (Option 2)
    - Row 3: Transparent surface + bold arrows (Option 4)
    - Left Column: Normal Obstacle
    - Right Column: Human Obstacle
    """
    fig = plt.figure(figsize=(16, 28))
    gs = gridspec.GridSpec(4, 2, width_ratios=[1, 1], height_ratios=[1, 0.8, 0.8, 0.8], wspace=0.12, hspace=0.35)
    
    # Obstacle parameters
    center = (IMAX // 2, JMAX // 2)  # center of grid
    obstacle_radius = 8   # cells - normal obstacle
    human_radius = 8      # cells - human radius matches obstacle for fair comparison
    
    # Create colormap for safety function
    colors = ['#1a1a2e', '#16213e', '#0f3460', '#533483', '#e94560']
    cmap = LinearSegmentedColormap.from_list('safety', colors[::-1])
    
    # Base scale for arrows - arrows will be scaled by dh0/base_dh0
    base_dh0 = DH0_OBSTACLE  # normalize relative to obstacle
    
    # configs: (title, is_human, apply_tangent, dh0, obs_color, radius)
    configs = [
        ('Normal Obstacle', False, False, DH0_OBSTACLE, '#888888', obstacle_radius),
        ('Human Obstacle (with Social Navigation)', True, True, DH0_HUMAN, 'red', human_radius),
    ]
    
    im = None
    
    # Create meshgrid for 3D plotting
    Y_grid, X_grid = np.mgrid[0:IMAX, 0:JMAX]
    X_grid = X_grid * DS
    Y_grid = Y_grid * DS
    
    for col_idx, (title, is_human, apply_tangent, dh0, obs_color, radius) in enumerate(configs):
        # Row 0: 2D plot
        ax_2d = fig.add_subplot(gs[0, col_idx])
        
        # Row 1: 3D plot (Surface only)
        ax_3d = fig.add_subplot(gs[1, col_idx], projection='3d')
        
        # Row 2: Flat plane projection (Option 2)
        ax_flat = fig.add_subplot(gs[2, col_idx], projection='3d')
        
        # Row 3: Transparent surface + bold arrows (Option 4)
        ax_trans = fig.add_subplot(gs[3, col_idx], projection='3d')
        
        # Create obstacle with the appropriate radius
        occ = create_circular_obstacle(center, radius)
        
        # Generate guidance field and safety function
        gx, gy, h, bound, boundary_cells, tangent_display = generate_guidance_and_safety(
            occ, is_human=is_human, apply_social_tangent=apply_tangent
        )
        
        # Arrow length scaling based on dh0
        arrow_scale_factor = (dh0 / base_dh0) * ARROW_VIS_SCALE
        
        # --- 2D PLOT (TOP) ---
        extent = [0, JMAX * DS, 0, IMAX * DS]
        im = ax_2d.imshow(h, origin='lower', cmap=cmap, vmin=0, vmax=1.2, extent=extent)
        
        # Plot obstacle
        circle = Circle((center[1] * DS, center[0] * DS), radius * DS,
                         color=obs_color, alpha=0.9, zorder=10)
        ax_2d.add_patch(circle)
        
        # Label
        label = 'H' if is_human else 'O'
        ax_2d.text(center[1] * DS, center[0] * DS, label,
                fontsize=16, ha='center', va='center', color='white',
                fontweight='bold', zorder=11)
        
        # Collect arrows for 3D plot
        arrows_3d_data = []
        
        # Better palette for "Closest = Deepest Color"
        # Unified Palette: White (Barrier) -> Gold -> Orange -> Light Orange
        if COLOR_ARROWS_BY_DEPTH:
             layer_colors = ['white', '#FFD700', '#FFA500', '#FFCC80', '#FFE0B2', 'cyan']
        else:
             layer_colors = ['white'] * 10
        
        # Number of arrows per ring (controls angular spacing)
        n_arrows_per_ring = 16
        
        # Spacing between layers in cells
        layer_spacing = 5
        
        # Determine loop range based on type
        # Human: layers 0..SOCIAL_TANGENT_LAYERS (remove outer normal) -> range(SOCIAL_TANGENT_LAYERS + 1)
        # Normal: layers 0..4 (remove outer) -> range(5)
        if apply_tangent:
            loop_range = range(SOCIAL_TANGENT_LAYERS + 1)
        else:
            loop_range = range(5)

        for layer in loop_range:
            # Radial distance for this layer
            layer_radius = radius + layer * layer_spacing + 1
            
            # Sample points
            angles = np.linspace(0, 2 * np.pi, n_arrows_per_ring, endpoint=False)
            
            # Pick color
            if COLOR_ARROWS_BY_DEPTH:
                idx = min(layer, len(layer_colors) - 1)
                c = layer_colors[idx]
            else:
                c = 'white'
            
            for angle in angles:
                # Grid position
                i = int(center[0] + layer_radius * np.cos(angle))
                j = int(center[1] + layer_radius * np.sin(angle))
                
                # Bounds check
                if not (1 <= i < IMAX - 1 and 1 <= j < JMAX - 1):
                    continue
                if bound[i, j] < 0:
                    continue
                
                # Compute radial outward direction
                di = i - center[0]
                dj = j - center[1]
                r = np.sqrt(di**2 + dj**2)
                if r < 0.01:
                    continue
                nx = di / r
                ny = dj / r
                
                # Compute arrow direction
                gx_val = nx
                gy_val = ny
                
                if apply_tangent and layer <= SOCIAL_TANGENT_LAYERS and layer > 0:
                    # Layers 1-N: Apply tangent bias
                    tx = -ny
                    ty = nx
                    decay = 1.0 - (layer - 1) / SOCIAL_TANGENT_LAYERS
                    sign = -1.0
                    biased_x = nx + sign * SOCIAL_TANGENT_BIAS * decay * tx
                    biased_y = ny + sign * SOCIAL_TANGENT_BIAS * decay * ty
                    bm = np.sqrt(biased_x**2 + biased_y**2)
                    gx_val = biased_x / bm
                    gy_val = biased_y / bm
                
                # Plot 2D arrow
                ax_2d.quiver(j * DS, i * DS, gy_val * arrow_scale_factor, gx_val * arrow_scale_factor,
                          color=c, alpha=0.9, scale=15, width=0.008,
                          zorder=5 + layer, headwidth=4, headlength=5)
                
                # Store for 3D
                # x (m), y (m), h_val, u, v, color
                x_m = j * DS
                y_m = i * DS
                h_val = h[i, j]
                
                # Estimate W (slope)
                # Simple approximation: project vector onto surface gradient
                # Or just let them float.
                # Let's compute slope: h_next - h_curr?
                # Using grid gradient:
                next_i = int(i + gx_val * 2) # check 2 cells away?
                next_j = int(j + gy_val * 2)
                next_i = np.clip(next_i, 0, IMAX-1)
                next_j = np.clip(next_j, 0, JMAX-1)
                h_next = h[next_i, next_j]
                
                # The arrow length in 3D is tricky. We'll use quiver length.
                arrows_3d_data.append((x_m, y_m, h_val, gy_val, gx_val, h_next - h_val, c))

        ax_2d.set_xlim(0, JMAX * DS)
        ax_2d.set_ylim(0, IMAX * DS)
        ax_2d.set_xlabel('X (m)', fontsize=12)
        ax_2d.set_ylabel('Y (m)', fontsize=12)
        ax_2d.set_title(title, fontsize=13, fontweight='bold')
        ax_2d.set_aspect('equal')
        
        # Legend
        if apply_tangent:
            if COLOR_ARROWS_BY_DEPTH:
                legend_text = (f'Layer 0 (White): Normal only\n'
                               f'Layer 1 (Gold): Max tangent\n'
                               f'Layer 2 (Orange): Decayed tangent\n'
                               f'Layer 3 (Light): Min tangent')
            else:
                legend_text = (f'Tangents applied (Layers 1-{SOCIAL_TANGENT_LAYERS})')
        else:
            if COLOR_ARROWS_BY_DEPTH:
                legend_text = (f'Arrows colored by depth\n'
                               f'(White=Close -> Gold -> Light)\n'
                               f'dh0 = {dh0}')
            else:
                legend_text = f'Normal gradient, dh0={dh0}'

        ax_2d.text(0.5, -0.15, legend_text,
                transform=ax_2d.transAxes, fontsize=10, va='top', ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

        # --- ROW 1: 3D SURFACE ONLY ---
        ax_3d.plot_surface(X_grid, Y_grid, h, cmap=cmap, linewidth=0, antialiased=False, vmin=0, vmax=1.2)
        ax_3d.set_title(f'3D Potential Field h(x) (dh0={dh0})', fontsize=12)
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('h(x)')
        ax_3d.set_zlim(0, 1.2)
        ax_3d.view_init(elev=35, azim=-135)
        
        # --- ROW 2: FLAT PLANE PROJECTION (Option 2) ---
        # Show surface with reduced opacity, arrows on a flat plane at z=1.15
        ax_flat.plot_surface(X_grid, Y_grid, h, cmap=cmap, linewidth=0, antialiased=False, vmin=0, vmax=1.2, alpha=0.5)
        ax_flat.set_title('Arrows Projected on Flat Plane (z=1.15)', fontsize=12)
        ax_flat.set_xlabel('X (m)')
        ax_flat.set_ylabel('Y (m)')
        ax_flat.set_zlabel('h(x)')
        ax_flat.set_zlim(0, 1.4)
        ax_flat.view_init(elev=35, azim=-135)
        
        # Plot arrows on flat plane z=1.15 (above the entire surface)
        flat_z = 1.15
        for (x, y, z_orig, u, v, w, c) in arrows_3d_data:
            ax_flat.quiver(x, y, flat_z, u, v, 0, color=c, length=0.35, normalize=True, arrow_length_ratio=0.3, linewidth=1.8)
        
        # --- ROW 3: TRANSPARENT SURFACE + BOLD ARROWS (Option 4) ---
        ax_trans.plot_surface(X_grid, Y_grid, h, cmap=cmap, linewidth=0, antialiased=False, vmin=0, vmax=1.2, alpha=0.3)
        ax_trans.set_title('Transparent Surface + Bold Arrows', fontsize=12)
        ax_trans.set_xlabel('X (m)')
        ax_trans.set_ylabel('Y (m)')
        ax_trans.set_zlabel('h(x)')
        ax_trans.set_zlim(0, 1.2)
        ax_trans.view_init(elev=35, azim=-135)
        
        # Plot bold arrows on top of transparent surface
        z_offset = 0.08
        for (x, y, z_orig, u, v, w, c) in arrows_3d_data:
            # Use darker outline for visibility against transparent surface
            bold_color = '#1a1a1a' if c == 'white' else c
            ax_trans.quiver(x, y, z_orig + z_offset, u, v, 0, color=bold_color, length=0.45, normalize=True, arrow_length_ratio=0.4, linewidth=2.5)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Poisson Safety Function h(x) [m]', fontsize=11)
    
    plt.suptitle('Guidance Field Modulation: Normal Obstacle vs Human with Social Navigation', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    output_path = os.path.join(SCRIPT_DIR, 'guidance_field_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def plot_layer_diagram():
    """
    Diagram showing the layer structure around a human.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#f8f8f8')
    
    center = (0, 0)
    human_radius = 0.4
    
    # Human
    human = Circle(center, human_radius, color='red', alpha=0.9, zorder=10)
    ax.add_patch(human)
    ax.text(0, 0, 'Human', fontsize=12, ha='center', va='center', 
            color='white', fontweight='bold', zorder=11)
    
    # Layer rings
    layer_colors = ['#FFD700', '#FFA500', '#FF6347', '#FF4500']
    cell_size = DS
    
    for layer in range(0, SOCIAL_TANGENT_LAYERS + 1):
        r_inner = human_radius + layer * cell_size * 2
        r_outer = human_radius + (layer + 1) * cell_size * 2
        
        if layer == 0:
            color = '#00CED1'  # Cyan for layer 0 (boundary)
            label = 'Layer 0\n(Boundary)\nNORMAL ONLY'
        else:
            color = layer_colors[min(layer - 1, len(layer_colors) - 1)]
            label = f'Layer {layer}\n+Tangent'
        
        # Draw ring
        ring = plt.Circle(center, r_outer, fill=False, color=color, 
                         linewidth=3, linestyle='--', zorder=5)
        ax.add_patch(ring)
        
        # Label
        angle = np.pi / 4 + layer * np.pi / 8
        label_r = (r_inner + r_outer) / 2
        lx = label_r * np.cos(angle)
        ly = label_r * np.sin(angle)
        ax.text(lx, ly, label, fontsize=9, ha='center', va='center',
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Show example vectors
    # Layer 0: normal only
    r0 = human_radius + cell_size
    theta0 = -np.pi / 6
    pos0 = (r0 * np.cos(theta0), r0 * np.sin(theta0))
    normal0 = (np.cos(theta0), np.sin(theta0))
    ax.annotate('', xy=(pos0[0] + 0.15 * normal0[0], pos0[1] + 0.15 * normal0[1]),
               xytext=pos0, arrowprops=dict(arrowstyle='->', color='cyan', lw=3))
    ax.text(pos0[0] + 0.2 * normal0[0], pos0[1] + 0.2 * normal0[1] - 0.05,
            'g', fontsize=12, color='cyan', fontweight='bold')
    
    # Layer 1: normal + tangent
    r1 = human_radius + 3 * cell_size
    theta1 = -np.pi / 6
    pos1 = (r1 * np.cos(theta1), r1 * np.sin(theta1))
    normal1 = (np.cos(theta1), np.sin(theta1))
    tangent1 = (-np.sin(theta1), np.cos(theta1))  # CCW perpendicular
    
    # Normal component
    ax.annotate('', xy=(pos1[0] + 0.12 * normal1[0], pos1[1] + 0.12 * normal1[1]),
               xytext=pos1, arrowprops=dict(arrowstyle='->', color='green', lw=2))
    # Tangent component
    ax.annotate('', xy=(pos1[0] + 0.08 * tangent1[0] * -1, pos1[1] + 0.08 * tangent1[1] * -1),
               xytext=pos1, arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    # Resultant
    result1 = (normal1[0] - 0.5 * tangent1[0], normal1[1] - 0.5 * tangent1[1])
    mag = np.sqrt(result1[0]**2 + result1[1]**2)
    result1 = (result1[0] / mag, result1[1] / mag)
    ax.annotate('', xy=(pos1[0] + 0.18 * result1[0], pos1[1] + 0.18 * result1[1]),
               xytext=pos1, arrowprops=dict(arrowstyle='->', color='yellow', lw=3))
    ax.text(pos1[0] + 0.22 * result1[0], pos1[1] + 0.22 * result1[1],
            "g'", fontsize=12, color='darkorange', fontweight='bold')
    
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.8, 1.0)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title('Layer Structure for Social Navigation\n'
                 '(Boundary preserves CBF normal, Layers 1+ add tangent)', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.4)
    
    # Legend box
    textstr = (
        "Layer 0 (Boundary): Pure normal g\n"
        "  → Preserves CBF safety guarantee\n\n"
        f"Layers 1-{SOCIAL_TANGENT_LAYERS}: g + β·g_⊥\n"
        f"  → Tangent bias β={SOCIAL_TANGENT_BIAS}\n"
        "  → Robot curves left (passes on human's right)\n\n"
        "g = normal gradient (cyan/green)\n"
        "g_⊥ = CCW tangent (orange)\n"
        "g' = biased guidance (yellow)"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    output_path = os.path.join(SCRIPT_DIR, 'layer_structure_diagram.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    print("Generating guidance field and Poisson safety function visualizations...")
    print(f"Grid: {IMAX}x{JMAX}, Cell size: {DS}m")
    print(f"dh0_obstacle: {DH0_OBSTACLE}, dh0_human: {DH0_HUMAN}")
    print(f"Social tangent: layers={SOCIAL_TANGENT_LAYERS}, bias={SOCIAL_TANGENT_BIAS}")
    print()
    
    plot_comparison()
    plot_layer_diagram()

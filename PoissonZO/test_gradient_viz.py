#!/usr/bin/env python3
"""
Minimal script to visualize guidance field with two circles:
- Circle 1: class_map = 1 (human) 
- Circle 2: class_map = 3 (obstacle)
User can move them closer/further with a slider.
"""

import sys
sys.path.insert(0, '/home/yangl/semantic-safety/PoissonZO/build')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import poisson_solver as ps

# Grid setup
IMAX = 120
JMAX = 120
DS = 0.05

ps.set_IMAX(IMAX)
ps.set_JMAX(JMAX)
ps.set_DS(DS)
ps.set_dh0_human(3.0)
ps.set_dh0_obstacle(0.3)
ps.set_robot_length(0.5)
ps.set_robot_width(0.3)

def create_circle_occ_and_class(center1, radius1, class1, center2, radius2, class2):
    """Create occupancy grid and class map with two circles."""
    occ = np.zeros(IMAX * JMAX, dtype=np.float32)
    class_map = np.zeros(IMAX * JMAX, dtype=np.int32)
    
    for i in range(IMAX):
        for j in range(JMAX):
            idx = i * JMAX + j
            
            # Distance to center1
            d1 = np.sqrt((i - center1[0])**2 + (j - center1[1])**2)
            if d1 <= radius1:
                occ[idx] = -1.0  # Occupied
                class_map[idx] = class1
            
            # Distance to center2
            d2 = np.sqrt((i - center2[0])**2 + (j - center2[1])**2)
            if d2 <= radius2:
                occ[idx] = -1.0  # Occupied
                class_map[idx] = class2
    
    # Free space = 1.0
    occ[occ == 0] = 1.0
    
    return occ, class_map

def solve_and_get_fields(separation):
    """Solve Poisson and return h-field and guidance field."""
    # Fixed circle (human) at center-left
    center1 = (IMAX // 2, JMAX // 2 - int(separation / 2))
    radius1 = 8
    class1 = 1  # Human
    
    # Moving circle (obstacle) at center-right
    center2 = (IMAX // 2, JMAX // 2 + int(separation / 2))
    radius2 = 8
    class2 = 3  # Obstacle
    
    occ, class_map = create_circle_occ_and_class(center1, radius1, class1, 
                                                   center2, radius2, class2)
    
    # Solve Poisson
    h_grid, gx, gy, iters = ps.solve_poisson_safety_function(occ, 0.0, class_map)
    
    # Reshape to 2D
    h_2d = h_grid.reshape(IMAX, JMAX)
    gx_2d = gx.reshape(IMAX, JMAX)
    gy_2d = gy.reshape(IMAX, JMAX)
    occ_2d = occ.reshape(IMAX, JMAX)
    class_2d = class_map.reshape(IMAX, JMAX)
    
    return h_2d, gx_2d, gy_2d, occ_2d, class_2d, iters

# Initial solve
init_separation = 30
h_2d, gx_2d, gy_2d, occ_2d, class_2d, iters = solve_and_get_fields(init_separation)

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plt.subplots_adjust(bottom=0.2)

# H-field plot
im_h = axes[0].imshow(h_2d, origin='lower', cmap='hot')
axes[0].set_title(f'H-field (iters={iters})')
plt.colorbar(im_h, ax=axes[0])

# Class map overlay
im_class = axes[1].imshow(class_2d, origin='lower', cmap='tab10', vmin=0, vmax=10)
axes[1].set_title('Class Map (1=human, 3=obstacle)')
plt.colorbar(im_class, ax=axes[1])

# Guidance field quiver
stride = 6
Y, X = np.mgrid[0:IMAX:stride, 0:JMAX:stride]
U = gy_2d[::stride, ::stride]  # j-direction = x in plot
V = gx_2d[::stride, ::stride]  # i-direction = y in plot
quiver = axes[2].quiver(X, Y, U, V, scale=20)
axes[2].imshow(occ_2d, origin='lower', alpha=0.3, cmap='gray')
axes[2].set_title('Guidance Field')
axes[2].set_xlim(0, JMAX)
axes[2].set_ylim(0, IMAX)

# Slider
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Separation', 10, 60, valinit=init_separation, valstep=1)

def update(val):
    separation = int(slider.val)
    h_2d, gx_2d, gy_2d, occ_2d, class_2d, iters = solve_and_get_fields(separation)
    
    # Update H-field
    im_h.set_data(h_2d)
    im_h.set_clim(h_2d.min(), h_2d.max())
    axes[0].set_title(f'H-field (iters={iters})')
    
    # Update class map
    im_class.set_data(class_2d)
    
    # Update quiver
    U = gy_2d[::stride, ::stride]
    V = gx_2d[::stride, ::stride]
    quiver.set_UVC(U, V)
    
    # Update occupancy overlay
    axes[2].images[0].set_data(occ_2d)
    
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.suptitle('Poisson Safety Function: Human (class=1) vs Obstacle (class=3)')
plt.show()

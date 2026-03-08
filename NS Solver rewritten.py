# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 12:47:23 2025

@author: Owner
"""

"The Navier-Stokes Solver is being rewritten to help with readability and redundancy"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"Constants"

nx, ny = 41, 41    # grid points in x and y
dx = 1/(nx-1)      
dy = 1/(ny-1)
nit = 1000     # number of iterations for PPE solver

rho = 1        #density of the fluid
nu = 1         #fluid's kinematic viscosity
#no physical units on any of these values
F = 0             #physical body forces acting on the fluid 

x = np.linspace(0, (nx - 1) * dx, nx)
y = np.linspace(0, (ny - 1) * dy, ny)
X, Y = np.meshgrid(x, y)                    #grid for fluid modelling


"Debugging Functions"

def calculate_divergence(r, s, dx, dy):
    div = d_dx_safe(r, dx) + d_dy_safe(s, dy)        
    return div




"Useful approximations"

#central difference theorem

def d_dx_safe(f, dx):
    """First derivative wrt x (columns). Returns array same shape as f."""
    g = np.zeros_like(f)
    # central differences for interior in x (axis=1)
    g[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * dx)
    return g

def d_dy_safe(f, dy):
    """First derivative wrt y (rows). Returns array same shape as f."""
    g = np.zeros_like(f)
    # central differences for interior in y (axis=0)
    g[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2.0 * dy)
    return g

def d2_dx2_safe(f, dx):
    """Second derivative wrt x (columns)."""
    g = np.zeros_like(f)
    g[:, 1:-1] = (f[:, 2:] - 2.0 * f[:, 1:-1] + f[:, :-2]) / (dx * dx)
    return g

def d2_dy2_safe(f, dy):
    """Second derivative wrt y (rows)."""
    g = np.zeros_like(f)
    g[1:-1, :] = (f[2:, :] - 2.0 * f[1:-1, :] + f[:-2, :]) / (dy * dy)
    return g


"Functions"

def check_sanity(u, v, p, n):
    if np.isnan(u).any() or np.isnan(v).any() or np.isnan(p).any():
        raise RuntimeError(f"NaNs detected at timestep {n}")
    if np.max(np.abs(u)) > 1e10 or np.max(np.abs(v)) > 1e10:
        raise RuntimeError(f"Velocity overflow at timestep {n}")
#ensure no NaNs/other diverging to infinity/divide by 0 errors
        
        
def compute_stable_dt(u, v, dx, dy, nu, safety=0.5):

    # Maximum velocity magnitude (avoid division by zero)
    umax = np.max(np.abs(u))
    vmax = np.max(np.abs(v))
    vel_max = max(umax, vmax, 1e-12)  # small number avoids zero division
    
    # safety (advection) constraint
    dt_adv = safety * min(dx, dy) / vel_max
    
    # Diffusion constraint
    dt_diff = safety * dx**2 * dy**2 / (2 * nu * (dx**2 + dy**2))
    
    return min(dt_adv, dt_diff)

def apply_velocity_boundary_cndts(u, v):
    
    #apply boundary cndts not incl. corners
    
    u[0, 1:-1] = 0             #bottom
    u[-1, 1:-1] = 1            #top
    u[1:-1, 0] = 0             #left
    u[1:-1, -1] = 0            #right

    v[0, 1:-1] = 0
    v[-1, 1:-1] = 0
    v[1:-1, 0] = 0
    v[1:-1, -1] = 0
    
    #apply corner cndts
    
    u[0, 0] = 0         #bottom left
    u[0, -1] = 0        #bottom right
    u[-1, 0] = 0        #top left
    u[-1, -1] = 0       #top right
    
    return u, v
        
def calculate_intermediate_velocities(u, dx, v, dy, nu, dt, solid):
    un = u.copy()           #create copies so dont run into issues editing 
                            #then trying to pull from original data
    vn = v.copy()

    u_star = un.copy()
    v_star = vn.copy()

    # compute derivatives without mutating un/vn
    du_dx = d_dx_safe(un, dx)
    du_dy = d_dy_safe(un, dy)
    dv_dx = d_dx_safe(vn, dx)
    dv_dy = d_dy_safe(vn, dy)

    lap_u = d2_dx2_safe(un, dx) + d2_dy2_safe(un, dy)
    lap_v = d2_dx2_safe(vn, dx) + d2_dy2_safe(vn, dy)

    # update interior and non walls only
    
    fluid = ~solid

    mask = fluid.copy()
    mask[0,:] = mask[-1,:] = mask[:,0] = mask[:,-1] = False     
    #this ensures only interior points are updated with mask
    
    u_star[mask] = (un[mask] +
                    dt * ( - (un[mask] * du_dx[mask])
                           - (vn[mask] * du_dy[mask])
                           + nu * lap_u[mask] ))
    
    v_star[mask] = (vn[mask] +
                    dt * ( - (un[mask] * dv_dx[mask])
                           - (vn[mask] * dv_dy[mask])
                           + nu * lap_v[mask] ))
    
    return u_star, v_star

def build_b(u_star, v_star, dx, dy, rho, dt, solid):   #intermediate step for pressure solve
                                                #b is a constant equal to both sides 
                                                #of solved NS eqn
    du_dx = d_dx_safe(u_star, dx)
    dv_dy = d_dy_safe(v_star, dy)
    b = (rho / dt) * (du_dx + dv_dy)
    b[solid] = 0
    return b 
    
def solve_PPE(p, b, dx, dy, nit, solid):       #gives pressure field
    p = p.copy()
    
    for _ in range (nit):
        pn = p.copy()
        fluid = ~solid
        mask = fluid.copy()
        mask[0,:] = mask[-1,:] = mask[:,0] = mask[:,-1] = False
        
        # Compute interior pressure update
        rhs = (
            ((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
             (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2 -
             b[1:-1, 1:-1] * dx**2 * dy**2)
            / (2.0 * (dx**2 + dy**2))
        )
        
        # Apply mask only to interior region
        p_inner = p[1:-1, 1:-1]
        mask_inner = mask[1:-1, 1:-1]
        
        p_inner[mask_inner] = rhs[mask_inner]
        
        p[1:-1, 1:-1] = p_inner
        
        # fill solid pressures from neighbours
        #needed as was having issue with fluid going into solid due to
        #pressure inside solid = 0
        p_avg = (
            np.roll(p,1,0) +
            np.roll(p,-1,0) +
            np.roll(p,1,1) +
            np.roll(p,-1,1)
        ) / 4
        
        p[solid] = p_avg[solid]

        # Neumann BCs: dp/dx = 0 and dp/dy = 0
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]


        p[0, 0] = 0.0  # anchor
    return p
    
def correct_velocities(u_star, v_star, p, dx, dy, rho, dt, solid):
    dp_dx = d_dx_safe(p, dx)
    dp_dy = d_dy_safe(p, dy)
    u = u_star.copy()
    v = v_star.copy()
    
    fluid = ~solid
    mask = fluid.copy()
    
    mask[0,:] = mask[-1,:] = mask[:,0] = mask[:,-1] = False
    u[mask] = u_star[mask] - (dt / rho) * dp_dx[mask]
    v[mask] = v_star[mask] - (dt / rho) * dp_dy[mask]
    return u, v

def create_cylinder(X, Y, cx, cy, r):               #for adding in walls/shapes
    return (X - cx)**2 + (Y - cy)**2 <= r**2

def apply_solid(u, v, solid):     #ensures velocity and pressure = 0 in wall
    u[solid] = 0
    v[solid] = 0

    return u, v
    
"Initial conditions"

u = np.zeros((ny,nx))
v = np.zeros((ny,nx))
p = np.zeros((ny,nx))          
b = np.zeros((ny,nx))

"Running code"

nt = 70        #how long simulation lasts for
snapshots = []

solid = create_cylinder(X, Y, 0.5, 0.5, 0.1)

for i in range(nt+1):
    
    dt = compute_stable_dt(u, v, dx, dy, nu, safety=0.5)
    u, v = apply_velocity_boundary_cndts(u, v)

    #step 1: intermediate velocities
    
    u_star, v_star = calculate_intermediate_velocities(u, dx, v, dy, nu, dt, solid)
    u_star, v_star = apply_velocity_boundary_cndts(u_star, v_star)
    u_star, v_star = apply_solid(u_star, v_star, solid)
    
    #step 2: calculate pressure field
    
    b = build_b(u_star, v_star, dx, dy, rho, dt, solid)
    p = solve_PPE(p, b, dx, dy, nit, solid)
    u_star, v_star = apply_solid(u_star, v_star, solid)
    
    #step 3: calculate final velocities from pressure field
    
    u, v = correct_velocities(u_star, v_star, p, dx, dy, rho, dt, solid)
    u, v = apply_velocity_boundary_cndts(u, v)
    u, v = apply_solid(u, v, solid)
    
    check_sanity(u, v, p, i)
    
    if i % 5 == 0:  # store every 2nd frame to reduce memory
        snapshots.append((u.copy(), v.copy(), p.copy()))


"Creating animation"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

# Create coordinate grids for plotting
x = np.linspace(0, (nx - 1) * dx, nx)
y = np.linspace(0, (ny - 1) * dy, ny)
X, Y = np.meshgrid(x, y)

def animate(frame):
    u_frame, v_frame, p_frame = snapshots[frame]


    ax1.clear()
    ax1.set_title("Pressure Field")
    ax1.contourf(X, Y, p_frame, 70, cmap='viridis')
    ax1.contour(X, Y, solid, levels=[0.5], colors='black')

    ax2.clear()
    ax2.set_title("Velocity field")
    speed = np.sqrt(u_frame**2 + v_frame**2)
    ax2.quiver(X, Y, u_frame, v_frame, speed, cmap='plasma')
    ax2.contour(X, Y, solid, levels=[0.5], colors='black')




ani = FuncAnimation(fig, animate, frames=len(snapshots), interval=50, blit = False)
ani.save("fluid_simulation.mp4", writer='ffmpeg', fps=3)
print("Animation saved to fluid_simulation.mp4")
    